import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, Self


class _Record(Protocol):
    """
    Structural contract for stage artifacts that can be serialized and keyed.
    """

    @property
    def record_key(self) -> int: ...

    def model_dump(self, *, mode: str = ...) -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, obj: object) -> Self: ...


def read_records[RecordT: _Record](path: Path, model: type[RecordT]) -> Iterator[RecordT]:
    """
    Stream records from a JSONL file, validating each line against the model.

    :param path: Source JSONL file.
    :param model: Model the records are validated against.
    :return: Iterator of validated records.
    """
    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            yield model.model_validate(json.loads(line))


def completed_keys[RecordT: _Record](path: Path, model: type[RecordT]) -> set[int]:
    """
    Collect the keys of records already written to a stage output for resume.

    :param path: Output JSONL file, possibly absent or truncated mid-write.
    :param model: Model the existing records are validated against.
    :return: Keys of the records already present in the file.
    """
    if not path.exists():
        return set()

    lines = path.read_text(encoding="utf-8").splitlines()
    keys: set[int] = set()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        keys.add(model.model_validate(data).record_key)
    return keys


@contextmanager
def append_writer[RecordT: _Record](path: Path) -> Iterator[Callable[[RecordT], None]]:
    """
    Open a JSONL file for appending and yield a per-record writer that flushes.

    :param path: Destination JSONL file.
    :return: Writer appending one flushed line per record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:

        def write(record: RecordT) -> None:
            fp.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            fp.write("\n")
            fp.flush()

        yield write
