# ClinicalFineSurE
Fine-grained Evaluation of Clinical Dialogue Summarization Task 

## Quick Start Guide
### 1. Clone Base Dataset (MTS-Dialog) Repos & Pre-process
```sh
# from project directory
cd ..

# clone & copy MTS-Dialog dataset sample
git clone https://github.com/abachaa/MTS-Dialog.git
cp MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv ClinicalFineSurE/dataset/original/

# Random Sample portion from MTS-Dialog dataset
mlr --icsv --ocsv sample -k 10 ClinicalFineSurE/dataset/original/MTS-Dialog-TrainingSet.csv > ClinicalFineSurE/dataset/sampled/MTS-Dialog.csv

# pre-process
cd ClinicalFineSurE
python -m preprocess.pseudo-labeling
```
