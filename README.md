# GNNMERGE: Merging of GNN Models Without Accessing Training Data

This is the PyTorch implementation for GNNMERGE: Merging of GNN Models Without Accessing Training Data

## Downloading Datasets

For the datasets used for in-domain model merging experiments, you need to use `download_datasets_in-domain.py` as follows:
```
python3 --dataset <dataset_name> --save_path <path_to_save_dataset>
```

For the datasets used for cross-domain and different tasks model merging, you can download the datasets from the following links:

**Citeseer** : https://anonymfile.com/JEW9O/citeseer-fixedsbert.pt

**Pubmed** : https://anonymfile.com/6NKk9/pubmed-fixedsbert.pt

**WikiCS** : https://anonymfile.com/AWoqY/wikics-fixedsbert.pt

**Cora** : https://anonymfile.com/La5xo/cora-fixedsbert.pt

**Arxiv** : https://anonymfile.com/Vpr3q/ogbn-arxivfixedsbert.pt

The `.pt` files can be directly used by the training and merging scripts if passed as dataset paths.
## Setup

After downloading the repository and entering into it, run:
```
conda env create -f environment.yml
conda activate gnnmerge
``` 
to set up and activate the conda environment for the experiments.

## Training Models

In this section, we describe how to train the models for various experimental settings.
### In-domain

For in-domain models, we create labelsplits of the same dataset. The label splitting is handled by the training script. Run the following command to train 2 models trained on disjoint label splits:

```
cd In-domain
python3 train_labelsplits.py --dataset <dataset_name> --model <model_name> --data_path <path_to_dataset> --model1_save_path <path_to_save_model1> --model2_save_path <path_to_save_model2> --logs_path <path_to_save_logs>
```
### Cross-domain
For cross-domain models, we train models on seperate datasets. Here, you would need to use SentenceBERT datasets. The links for them have been provided in the **Download Datasets** section. To train a model for a particular dataset, run:
```
cd Cross-domain
python3 train_nc_model.py --dataset_name <dataset_name> --model_name <model_name> --data_path <path_to_dataset> --logs_path <path_to_save_logs> --model_save_path <path_to_save_model>
```

### Different Tasks
Here, you would need to use SentenceBERT datasets. The links for them have been provided in the **Download Datasets** section. To train a model for node classification, run:
```
cd Different_Tasks
python3 train_nc_model.py --dataset_name <dataset_name> --model_name <model_name> --data_path <path_to_dataset> --logs_path <path_to_save_logs> --model_save_path <path_to_save_model>
```
To train a model for link prediction, run:
```
cd Different_Tasks
python3 train_lp_model.py --dataset_name <dataset_name> --model_name <model_name> --data_path <path_to_dataset> --logs_path <path_to_save_logs> --model_save_path <path_to_save_model>
```
## Merging Models

### In-domain

To use **GNNMerge** to merge in-domain models, use:
```
python GNNMerge.py --dataset_name <dataset_name> --model_name <model_name> --data_path <path_to_dataset> --model1_path <path_to_first_model> --model2_path <path_to_second_model> --logs_path <path_to_save_logs>
```
To use **GNNMerge++** to merge in-domain models, use:
```
python GNNMerge++.py --dataset_name <dataset_name> --model_name <model_name> --data_path <path_to_dataset> --model1_path <path_to_first_model> --model2_path <path_to_second_model> --logs_path <path_to_save_logs>
```

### Cross-domain

**Ensure that the order of the data paths, dataset names and model paths is consisent!**

To use **GNNMerge** to merge cross-domain models, use:
```
python GNNMerge.py --data_paths <path_to_dataset1> <path_to_dataset2> ... --dataset_names <dataset_name1> <dataset_name2> ... --model_paths <path_to_model1> <path_to_model2> ... --model_name <model_name> --logs_path <path_to_save_logs>
```

To use **GNNMerge++** to merge cross-domain models, use:
```
python GNNMerge++.py --data_paths <path_to_dataset1> <path_to_dataset2> ... --dataset_names <dataset_name1> <dataset_name2> ... --model_paths <path_to_model1> <path_to_model2> ... --model_name <model_name> --logs_path <path_to_save_logs>
```

### Different Tasks

To use **GNNMerge** to merge node classification and link prediction models, use:
```
python GNNMerge.py --nc_dataset_name <nc_dataset_name> --lp_dataset_name <lp_dataset_name> --model_name <model_name> --nc_data_path <path_to_nc_dataset> --lp_data_path <path_to_lp_dataset> --nc_model_path <path_to_nc_model> --lp_model_path <path_to_lp_model> --logs_path <path_to_save_logs>
```

To use **GNNMerge++** to merge node classification and link prediction models, use:
```
python GNNMerge++.py --nc_dataset_name <nc_dataset_name> --lp_dataset_name <lp_dataset_name> --model_name <model_name> --nc_data_path <path_to_nc_dataset> --lp_data_path <path_to_lp_dataset> --nc_model_path <path_to_nc_model> --lp_model_path <path_to_lp_model> --logs_path <path_to_save_logs>
```