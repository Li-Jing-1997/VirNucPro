# VirNucPro: An Identifier for the Identification of Viral Short Sequences Using Six-Frame Translation and Large Language Models

## Installation
Clone the current repo

    git clone https://github.com/Li-Jing-1997/VirNucPro.git

    conda create -n VirNucPro python=3.9
    conda activate VirNucPro

    pip install -r requirements.txt
    pip uninstall triton # this can lead to errors in GPUs other than A100

## Run VirNucPro for prediction
Prepare the sequence file you want to predict, and run:
```
python prediction.py input_sequence.fasta Expected_length model.pth
```
For example:
```
python prediction.py input_sequence.fasta 500 500_model.pth
```
The prediction results will be saved in a folder prefixed with input_sequence and suffixed with _merged. e.g., `input_sequence_merged/prediction_results.txt`

You can also visualize the prediction results using:
```
python drew_fig.py input_sequence_merged/prediction_results.txt
```

## Model training
1.Download data
You can also train your own model according to your task.
You can download the training data from [NCBI](https://ftp.ncbi.nlm.nih.gov/refseq/release/) using download_data.py
```
python download_data.py
```
This will download the data for viral and randomly select enough data from other species for downloading.

2.Prepare train dataset
Next, prepare the training dataset according to the expected prediction length, for 300bp:
```
python make_train_dataset_300.py
```
and for 500bp:
```
python make_train_dataset_500.py
```
Nucleic acid and corresponding amino acid sequences will stored with the suffixes identified_nucleotide.fa and identified_protein.fa, respectively.

3.Extract features
Extract features unsing DNABERT_S and ESM2_3B by running:
```
python features_extract.py
```
This will identify all files in the data folder with the names identified_nucleotide.fa and identified_protein.fa, extract all sequences that start with "viral," and randomly select an equal number of sequences from other species in proportionate amounts.

4.Train Model
Use train.py for training:
```
python train.py
```
The trained model will be saved to `model.pth`.