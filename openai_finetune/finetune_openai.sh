#!/bin/bash



openai tools fine_tunes.prepare_data -f train.jsonl -q

openai api fine_tunes.create -t train_prepared.jsonl -v valid_prepared.jsonl -m curie --suffix "test1"

openai api fine_tunes.follow -i <job-id>
openai api fine_tunes.get -i <JOB_ID>
openai api fine_tunes.list
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>

openai api fine_tunes.results -i ft-2zaA7qi0rxJduWQpdvOvmGn3 > result.csv

# For multiclass classification
openai api fine_tunes.create \
  -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_OR_PATH> \
  -m <MODEL> \
  --compute_classification_metrics \
  --classification_n_classes <N_CLASSES>

# For binary classification
openai api fine_tunes.create \
  -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_OR_PATH> \
  -m <MODEL> \
  --compute_classification_metrics \
  --classification_n_classes 2 \
  --classification_positive_class <POSITIVE_CLASS_FROM_DATASET>

openai api fine_tunes.follow -i ft-zcrJJXblwsGBOeldFV8BLBkP
openai api fine_tunes.results -i ft-zcrJJXblwsGBOeldFV8BLBkP
