# Code2vec - Transfer Learning for code2vec in context of scratch programs

## Requirements

- Java 17
- Python 3.8


## Initialization

 ```bash
 poetry env use 3.8
 poetry install
 ```


## Preprocessing

Download the used dataset from [here](any-link). Unpack these and modyfy some path variables in `preprocess-pure.py`. 
If you use a `poetry` environment change the parameter value from  `python` to `poetry run python`.
After that run 
```bash
bash preprocess-pure.py
```

This will preprocess the given data to a starting point for training.

## Creating pre-trained models
### Downloading pre-trained models
The pre-trained models trained on sprites can downloaded [here]()

### Training an own pre-trained model (optional)
Modify some configuration params the `train.sh` script. Then run

```bash
bash train.sh
```

## Testing scores of the pre-trained model
To test a pre-trained model just use following command: 
```bash
[poetry run ] python code2vec.py --load [path_to_the_model] --test [path_to_the_test_data_set]
```

## Fine-Tuning
To use Transfer Learning you can adjust the values of the `fine-tune.sh` script. 
Modify some params and you can start the transfer learning process.
Run
```bash
bash fine-tune.sh
```

## Testing fine-tuned models. To test one ore more checkpoints of a pre-trained model, run:
```bash
bash test-finetuned-checkpoints.sh [DATASETNAME] [MODEL_NAME] [test-set] [start-checkpoint] [endcheckpoint]
```
It will evaluate each stored checkpoint and saves all metrics in a `csv` file of the model.