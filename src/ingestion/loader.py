import csv
import sys
from collections.abc import Iterator
from pathlib import Path

from src.types.source import SourceDocument


def load_websites(csv_path: str) -> Iterator[SourceDocument]:
    """
    Stream rows from the websites CSV file.

    :param csv_path: Path to the CSV file.
    :return: Iterator over parsed rows.
    """
    path = Path(csv_path)
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            yield SourceDocument(
                source_id=row["web_id"],
                url=row.get("url") or None,
                title=row.get("title") or None,
                kind=row.get("kind") or None,
                text=text,
            )
