# Cross-Validation FeatureCloud App

## Description
A Cross Validation FeautureCloud app, creating local splits for a k-fold cross validation.

## Input
- data.csv containing the local data (columns: features; rows: samples)

## Output
- data.csv containing the original local data
- folder containing a subfolder for each split. The subfolder contains the train and test file for the split.

## Workflows
Can be combined with the following apps:
- Post: 
  - Various preprocessing apps (e.g. Normalization, Feature Selection, ...) 
  - Various analysis apps (e.g. Random Forest, Logistic Regression, Linear Regression)

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_cross_validation:
  input:
    data: "data.csv" # File containing the actual data
    label_column: "target" # Name of the column including the labels. Can be None if stratify is false
    sep: "," # Separator of the data file.
  output:
    train: "train.csv" # Output filename of the train set
    test: "test.csv" # Output filename of the test set
    split_dir: "data" # name of the dir including the splits
  cross_validation:
    n_splits: 10 # number of splits
    shuffle: true # Data will be shuffled before splits are created
    stratify: false # If true, labels will be equally distributed between the splits. If true, label_column cannot be Null
    random_state: Null # Seed to create reproducible splits, Null possible
```
