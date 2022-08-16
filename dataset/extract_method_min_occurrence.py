#! /usr/bin/env python3

import fileinput
import json
import sys


def get_frequencies() -> dict[str, int]:
    freq = {}

    for line in fileinput.FileInput(files="method_counts.txt"):
        splits = line.split()
        if len(splits) == 2:
            count, name = splits[0], splits[1]
            freq[name] = int(count)

    return freq


def process(line: str, freq: dict[str, int], min_freq: int) -> None:
    parsed = json.loads(line)
    name = parsed["label"]

    if name in freq.keys() and freq[name] >= min_freq:
        print(line)


def main(min_freq: int) -> None:
    freq = get_frequencies()
    for line in fileinput.input():
        process(line, freq, min_freq)


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    # min_freq = int(sys.argv[1])
    main(10)
