import os
import sys
import argparse
from os import walk
from pathlib import Path
import random


def main(args):
    # lines = []
    # for (dirpath, dirnames, filenames) in walk(args.path):
    #     for file_name in filenames:
    #         with open(Path(dirpath) / Path(file_name), 'r') as file:
    #             for line in file.readlines():
    #                 if(args.relabel):
    #                     tokens = line.split(" ")
    #                     name=Path(dirpath).name
    #                     name=name.replace(",","|")
    #                     tokens[0] = name
    #                     lines.append(' '.join(tokens))
    #                 else:
    #                     lines.append(line)
    projects = []
    for (dirpath, dirnames, filenames) in walk(args.path):
        for filename in filenames:
            projects.append((dirpath, filename))
    # print(projects)
    count = len(projects)
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
    for project in projects:
        type = set_type[i]
        if type == 0:
            train.append(project)
        elif type == 1:
            val.append(project)
        else:
            test.append(project)
        i += 1
        sys.stdout.write("\rTotal: {:.2f}%, Train: {:.2f}%, Val: {:.2f}%, Test {:.2f}%".format(i / count * 100,
                                                                                               len(train) / count * 100,
                                                                                               len(val) / count * 100,
                                                                                               len(test) / count * 100))
        sys.stdout.flush()
        if i == count:
            break
    print(f"\nSuccessfully collected {i} scratch files")

    print(train)
    print(test)
    print(val)
    sets = ["train", "validation", "test"]

    train_lines = relabel_lines_to_list(args, train)
    test_lines = relabel_lines_to_list(args, test)
    val_lines = relabel_lines_to_list(args, val)



    save(train_lines, "test", args)
    save(test_lines, "train",args)
    save(val_lines, "validation",args)


def relabel_lines_to_list(args, dataset):
    lines = []
    for (path, file) in dataset:
        with open(Path(path) / file, 'r') as f:
            for line in f.readlines():
                if args.relabel:
                    tokens = line.split(" ")
                    name = Path(path).name
                    name = name.replace(",", "|")
                    tokens[0] = name
                    lines.append(' '.join(tokens))
                else:
                    lines.append(line)
    return lines


def create_dataset_dir(dataset_name, args, category):
    os.makedirs(f"{args.output}/{dataset_name}/{category}", exist_ok=True)


def save(dataset, dataset_name, args):
    random.shuffle(dataset)
    with open(f"{args.output}/{dataset_name}.{args.ending}", 'w') as outfile:
        for line in dataset:
            outfile.write(line+"\n")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ending", type=str, default='c2v')
    parser.add_argument("--relabel", action='store_true')
    return parser


if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    main(args)
