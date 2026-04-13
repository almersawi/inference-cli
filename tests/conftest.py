from dataclasses import dataclass, field
from typing import Iterator
import pytest


@dataclass
class FakeDelta:
    content: str | None = None


@dataclass
class FakeChoice:
    delta: FakeDelta = field(default_factory=FakeDelta)


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class FakeChunk:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


class FakeStream:
    def __init__(self, chunks: list[FakeChunk], delays: list[float] | None = None):
        self._chunks = chunks
        self._delays = delays or [0.0] * len(chunks)

    def __iter__(self) -> Iterator[FakeChunk]:
        import time
        for chunk, delay in zip(self._chunks, self._delays):
            if delay:
                time.sleep(delay)
            yield chunk

    def close(self) -> None:
        pass


class FakeCompletions:
    def __init__(self, stream: FakeStream):
        self._stream = stream
        self.last_kwargs: dict | None = None

    def create(self, **kwargs) -> FakeStream:
        self.last_kwargs = kwargs
        return self._stream


class FakeChat:
    def __init__(self, stream: FakeStream):
        self.completions = FakeCompletions(stream)


class FakeClient:
    def __init__(self, stream: FakeStream):
        self.chat = FakeChat(stream)


def make_chunk(content: str | None = None, usage: FakeUsage | None = None) -> FakeChunk:
    choice = FakeChoice(delta=FakeDelta(content=content))
    return FakeChunk(choices=[choice], usage=usage)


@pytest.fixture
def fake_client_factory():
    def _factory(chunks: list[FakeChunk], delays: list[float] | None = None):
        return FakeClient(FakeStream(chunks, delays))
    return _factory


@pytest.fixture
def make_chunk_fn():
    return make_chunk


@pytest.fixture
def fake_usage_cls():
    return FakeUsage
