#! /usr/bin/env python3

import fileinput
import json
import sys


def process(line: str) -> None:
    parsed = json.loads(line)
    print(parsed["label"])


def main() -> None:
    for line in fileinput.input():
        process(line)


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    main()
