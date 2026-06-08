import csv
from collections.abc import Iterator
from itertools import islice
from pathlib import Path


def iter_questions(csv_path: Path, limit: int | None) -> Iterator[tuple[int, str]]:
    """
    Iterate question id and query pairs from a CSV file.

    :param csv_path: Path to a CSV with q_id and query columns.
    :param limit: Optional cap on how many rows to yield.
    :return: Iterator of question id and query pairs.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        source = ((int(row["q_id"]), row["query"]) for row in reader)
        if limit is not None:
            source = islice(source, limit)
        yield from source
