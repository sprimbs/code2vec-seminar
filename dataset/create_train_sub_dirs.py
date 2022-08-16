import os.path
import sys
from os import walk
from os.path import join
import random
import shutil


def get_all_files(path: str) -> [str]:
    res = []
    for (dir_path, dir_names, file_names) in walk(path):
        res.extend([dir_path + "/" + f for f in file_names])
    return res


def split(path: str) -> None:
    all_java_files = get_all_files(path)
    print(all_java_files)
    random.shuffle(all_java_files)
    training = all_java_files[:int(len(all_java_files) * 0.8)]
    validation = all_java_files[int(len(all_java_files) * 0.8):int(len(all_java_files) * 0.9)]
    testing = all_java_files[int(len(all_java_files) * 0.9):]
    for f in training:
        shutil.copy(f, join("./train", os.path.basename(f)))
    for f in validation:
        shutil.copy(f, join("./val", os.path.basename(f)))
    for f in testing:
        shutil.copy(f, join("./test", os.path.basename(f)))


if __name__ == "__main__":
    split(sys.argv[1])
