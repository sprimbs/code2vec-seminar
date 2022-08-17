# Code2vec

## Initialization
 ```
 poetry env use 3.8
 poetry install
 ```

After that find the path to the executable of the poetry interpreter. This path has to be set in the `preprocess.sh` and
`train.sh` in order to execute everything on the correct interpreter.

## Preprocessing
Move your `.java`-files or Java-repositories into `./dataset/sources` and execute 
```
poetry run python ./dataset/create_train_sub_dirs.py ./dataset/sources
```
This will split the data into training, validation and testing.

Now run 
```
bash preprocess.sh
```

## Training

Run 
```
bash train.sh
```
