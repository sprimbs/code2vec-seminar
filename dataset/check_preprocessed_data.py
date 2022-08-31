import sys


def check_preprocessed_data(file: str) -> None:
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if is_wrong_format(line):
                print(line)


def is_wrong_format(line: str) -> bool:
    split = line.split(" ")
    if split[0].count(",") > 0:
        return True
    for path in split[1:]:
        if path.count(",") > 2:
            return True
    return False


if __name__ == "__main__":
    check_preprocessed_data(sys.argv[1])
