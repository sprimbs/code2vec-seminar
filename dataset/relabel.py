import os
import sys
import argparse
from os import walk
from pathlib import Path
import random


def main(args):
    lines = []
    i = 0
    count = args.count if args.count is not None else len(lines)
    for (dirpath, dirnames, filenames) in walk(args.path):
        for file_name in filenames:
            with open(Path(dirpath) / Path(file_name), 'r') as file:

                for line in file.readlines():
                    if(args.relabel):
                        tokens = line.split(" ")
                        name=Path(dirpath).name
                        name=name.replace(",","|")
                        tokens[0] = name
                        lines.append(' '.join(tokens))
                    else:
                        lines.append(line)
                    i+=1
                    sys.stdout.write("\rTotal: {:.2f}%".format(i / count * 100 if count != 0 else i))
                if args.count is not None and i >= args.count:
                    break
    count = args.count if args.count is not None else len(lines)
    set_type = []
    for index in range(count):
        if index < 0.8 * count:
            set_type.append(0)
        elif index < 0.9 * count:
            set_type.append(1)
        else:
            set_type.append(2)
    random.shuffle(set_type)
    i = 0
    test = []
    train = []
    val = []
    for line in lines:
        type = set_type[i]
        if type == 0:
            train.append(line)
        elif type == 1:
            val.append(line)
        else:
            test.append(line)
        i += 1
        sys.stdout.write("\rTotal: {:.2f}%, Train: {:.2f}%, Val: {:.2f}%, Test {:.2f}%".format(i / count * 100,
                                                                                               len(train) / count * 100,
                                                                                               len(val) / count * 100,
                                                                                               len(test) / count * 100))
        sys.stdout.flush()
        if i == count:
            break
    print(f"\nSuccessfully collected {i} java files")
    save(test, "test", args)
    save(train, "train",args)
    save(val, "validation",args)


def save(dataset, dataset_name, args):
    random.shuffle(dataset)
    with open(f"{args.output}/{dataset_name}.{args.ending}", 'w') as outfile:
        for line in dataset:
            outfile.write(line)

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str,required=True)
    parser.add_argument("--ending", type=str,default='c2v')
    parser.add_argument("--relabel",action="store_true")
    return parser

if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    main(args)
