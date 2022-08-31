# Code2vec

## Requirements

- Java 17
- Python 3.8

## Cloning

```bash
git clone --recurse-submodules https://gitlab.infosun.fim.uni-passau.de/se2/deepcode/code2vec.git
```

## Initialization

 ```bash
 poetry env use 3.8
 poetry install
 ```

After that find the path to the executable of the poetry interpreter. This path has to be set in the `preprocess.sh` and
`train.sh` in order to execute everything on the correct interpreter.

Now the preprocessing-toolbox must be built, therefore run:

```bash
bash build_extractor.sh
```

## Preprocessing

Move your `.java`-files or Java-repositories into `./dataset/sources` and execute 

```bash
poetry run python ./dataset/create_train_sub_dirs.py ./dataset/sources
```

This will split the data into training, validation and testing.

Now run

```bash
bash preprocess.sh
```

## Training

Run

```bash
bash train.sh
```
