#!/usr/bin/env bash
###########################################################
# Change the following values to train a new model.
# type: the name of the new model, only affects the saved file name.
# dataset: the name of the dataset, as was preprocessed using preprocess.sh
# test_data: by default, points to the validation set, since this is the set that
#   will be evaluated after each training iteration. If you wish to test
#   on the final (held-out) test set, change 'val' to 'test'.
tag=""
type=project_category/pretrained/layer1
dataset_name=project_category
data_dir=data/${dataset_name}/$tag
data=${data_dir}/${dataset_name}
test_data=${data_dir}/${dataset_name}.val.c2v
model_dir=models/${type}
python=python


mkdir -p ${model_dir}
set -e
${python}  finetune_model.py --data ${data} --test ${test_data} --save ${model_dir}/saved_model   --pretrained-model models/sprite-models/model-600k/model-600k
