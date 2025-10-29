# Secure Disaster Tweets Classification – Intermediate Update 1

## Overview
This update presents the initial phase of the Disaster Tweets Classification project.  
The focus is on preprocessing the dataset, performing exploratory data analysis (EDA),  
and implementing baseline models to evaluate early performance.

## Dataset
The dataset is sourced from Kaggle – *Natural Language Processing with Disaster Tweets*.  
It includes labeled tweets to identify whether a message refers to a real disaster or not.  

Files used:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Implementation
Two models were implemented in this phase:

| Model | Accuracy | F1 Score |
|--------|-----------|-----------|
| TF-IDF + Logistic Regression | 0.8070 | 0.7637 |
| DistilBERT (2 epochs) | 0.8437 | 0.8128 |

## Outputs
The following outputs were generated and saved:
- Class distribution plots and tweet length histograms  
- Confusion matrices for baseline and transformer models  
- Performance metrics stored under `outputs/metrics/`  

## Challenges & Next Steps
- Fine-tuning transformer model for better generalization  
- Experimenting with additional text-cleaning techniques  
- Extending training to multiple epochs for performance gain  
- Preparing final comparative analysis for the next milestone
