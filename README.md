# FeatureCloud Cross-Validation app
The Cross-Validation app in the FeatureCloud library creates k-fold local splits, which can be used as one of the
first apps in FC workflow followed by different Machine Learning or Data analysis apps.

## Input
Similar to Data Distributor app, Cross-Validation (CV) accepts Numpy files (`.npy`, `.npz`) files alongside `.csv` and `.txt` files.
For `.csv` and `.txt` files, clients should identify the delimiter character using the `sep` key in the config file. For NumPy file 
they have to choose from three possible options for `target_value`:
- `same-sep`: target value in the same file but separate array, i.e., the first array includes samples and the second one contains target values.
  Evidently, both arrays should have the same length and corresponding features and labels for each sample.
```angular2html
f1, f2 = [1,2,3,4,5], [10,20,30,40,50]
l1, l2 = 0, 1
features = [f1, f2]
labels = [l1, l2]
dataset = [features, labels]
``` 

- `same-last`: like `same-sep` both labels and features are in the same file; however, each sample's label comes right
   at the end of array that contains features array. 
```angular2html
features = [1,2,3,4,5]
label = 0
sample = [features, label]
dataset = [sample, sample]
``` 
- Name a separate NumPy file (`.npy` and `.npz`) that contains target values.
## Output
The output directory includes splits of test and train data in the same format as the input file.

## Workflows
Can be combined with the following apps:
- Post: 
  - Various preprocessing apps (e.g. Normalization, Feature Selection, ...) 
  - Various analysis apps (e.g. Random Forest, Logistic Regression, Linear Regression)
![Diagram](../data/images/CrossValidation.png)
## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_cross_validation:
  local_dataset:
    data: data.npy
    target_value: same-sep
    sep: ','
  n_splits: 10
  shuffle: true
  stratify: false
  random_state: null
  split_dir: data
  result:
    train: train.npy
    test: test.npy
```
- `target_value` indicates where labels can be found inside the input data or can be the name of a different file in the 
same directory as the input data file, which contains labels. Beware that `target_value` should be a string, even if it's a number!
e.g., `target_value: '10'` 
- `n_splits`: number of splits to be created using CV.

## Requirements
- pandas
- numpy
- scikit-learn
