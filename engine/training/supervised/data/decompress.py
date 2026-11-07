import io
from itertools import islice
from pathlib import Path

import zstandard


#Reading the Lichess evals file: lichess_db_eval.jsonl.zst, one JSON position per line
#It's 22.5GB compressed and far bigger unpacked, so it's never unpacked to disk or loaded whole:
#it's decompressed a bit at a time as lines are read, and only X lines are held at once


#Opens the .zst file as text, decompressing on the fly as lines are read
def open_evals(path: Path) -> io.TextIOWrapper:
    raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
    return io.TextIOWrapper(raw, encoding="utf-8")


#Next x lines of the file, carrying on from where the last call stopped; [] once the file is used up
def read_lines(file: io.TextIOWrapper, x: int) -> list[str]:
    return list(islice(file, x))
