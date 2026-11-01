import asyncio
import pickle

import anyio
import pytest

import httpx


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, OSError, httpx.ReadError, KeyboardInterrupt, SystemExit],
)
async def test_aclose_retries_failed_cleanup(error_type):
    error = error_type("close failed")

    class Stream(httpx.AsyncByteStream):
        calls = 0
        closed = False

        async def aclose(self):
            self.calls += 1
            if self.calls == 1:
                raise error
            self.closed = True

    stream = Stream()
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(200, stream=stream, request=request)

    with pytest.raises(error_type) as caught:
        await response.aclose()
    assert caught.value is error
    if isinstance(error, httpx.RequestError):
        assert error.request is request
    assert response.is_closed
    with pytest.raises(httpx.StreamClosed):
        await response.aread()

    await response.aclose()
    assert stream.closed
    assert stream.calls == 2
    await response.aclose()
    assert stream.calls == 2


@pytest.mark.anyio
async def test_aclose_retries_cancelled_cleanup():
    entered = anyio.Event()
    scope = anyio.CancelScope()

    class Stream(httpx.AsyncByteStream):
        calls = 0
        closed = False

        async def aclose(self):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                await anyio.sleep_forever()
            self.closed = True

    stream = Stream()
    response = httpx.Response(200, stream=stream)

    async def close():
        with scope:
            await response.aclose()

    with anyio.fail_after(1):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(close)
            await entered.wait()
            scope.cancel()

    assert scope.cancelled_caught
    assert response.is_closed
    assert not stream.closed
    await response.aclose()
    assert stream.closed
    assert stream.calls == 2


@pytest.mark.anyio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_aclose_concurrent_calls(fail_first):
    entered = anyio.Event()
    release = anyio.Event()
    second_started = anyio.Event()
    errors = []

    class Stream(httpx.AsyncByteStream):
        calls = 0
        active = False
        closed = False

        async def aclose(self):
            assert not self.active
            self.active = True
            self.calls += 1
            try:
                if self.calls == 1:
                    entered.set()
                    await release.wait()
                    if fail_first:
                        raise RuntimeError("close failed")
                self.closed = True
            finally:
                self.active = False

    stream = Stream()
    response = httpx.Response(200, stream=stream)

    async def first_close():
        try:
            await response.aclose()
        except RuntimeError as exc:
            errors.append(exc)

    async def second_close():
        second_started.set()
        await response.aclose()

    with anyio.fail_after(1):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(first_close)
            await entered.wait()
            assert response.is_closed
            tasks.start_soon(second_close)
            await second_started.wait()
            release.set()

    assert len(errors) == int(fail_first)
    await response.aclose()
    assert stream.closed
    assert stream.calls == (2 if fail_first else 1)


@pytest.mark.anyio
async def test_aclose_reentrant():
    class Stream(httpx.AsyncByteStream):
        calls = 0

        async def aclose(self):
            self.calls += 1
            assert self.calls == 1
            await response.aclose()

    stream = Stream()
    response = httpx.Response(200, stream=stream)
    with anyio.fail_after(1):
        await response.aclose()
    assert response.is_closed
    assert stream.calls == 1


@pytest.mark.anyio
async def test_aclose_failed_response_remains_picklable():
    class Stream(httpx.AsyncByteStream):
        async def aclose(self):
            raise RuntimeError("close failed")

    response = httpx.Response(200, stream=Stream())
    with pytest.raises(RuntimeError, match="close failed"):
        await response.aclose()

    restored = pickle.loads(pickle.dumps(response))
    assert restored.is_closed
    await restored.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_client_retries_cancelled_response_cleanup(cancel_count):
    entered = [asyncio.Event() for _ in range(cancel_count)]

    class Stream(httpx.AsyncByteStream):
        calls = 0
        closed = False

        async def __aiter__(self):
            yield b"ok"

        async def aclose(self):
            self.calls += 1
            if self.calls <= cancel_count:
                entered[self.calls - 1].set()
                await asyncio.Event().wait()
            self.closed = True

    stream = Stream()
    response = httpx.Response(200, stream=stream)
    transport = httpx.MockTransport(lambda request: response)
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(client.get("https://example.org"))
        try:
            with anyio.fail_after(1):
                for checkpoint in entered:
                    await checkpoint.wait()
                    assert response.num_bytes_downloaded == 2
                    task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert response.is_closed
        assert stream.closed is (cancel_count == 1)
        await response.aclose()
        assert stream.closed
        assert stream.calls == cancel_count + 1
