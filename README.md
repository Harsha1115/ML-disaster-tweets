# Multi-Class Classification of Disaster-Related Tweets Using DistilBERT for Real-Time Emergency Response
### Intermediate Update 1

## Work Completed
- Loaded the Kaggle dataset: Natural Language Processing with Disaster Tweets.
- Cleaned missing values in the 'keyword' and 'location' columns.
- Performed Exploratory Data Analysis (EDA).
- Generated plots for class distribution and tweet length.
- Implemented two models:
  1) TF-IDF + Logistic Regression (Baseline Model)
  2) DistilBERT (Fine-tuned Transformer, 2 Epochs)
- Collected evaluation metrics and saved all results in organized folders.

## Outputs Generated
### 📊 EDA Plots
- class_balance.png  → Class distribution (0 = Non-Disaster, 1 = Disaster).
- tweet_length.png   → Tweet length histogram.

### 🤖 Model Performance
| Model | Accuracy | F1 Score |
|------|-----------|----------|
| TF-IDF + Logistic Regression | 0.8070 | 0.7637 |
| DistilBERT (2 epochs)        | 0.8437 | 0.8128 |

### 📈 Evaluation Figures
- baseline_cm.png   → Confusion matrix (Baseline).
- distilbert_cm.png → Confusion matrix (DistilBERT).

### 🗂️ Metrics Files
- baseline.csv  → Accuracy and F1 of Logistic Regression.
- distilbert.csv → Accuracy and F1 of DistilBERT model.

### 📁 Folder Layout
```
ml-m1-disaster-tweets/
│
├── data/raw/ → train.csv, test.csv, sample_submission.csv
├── outputs/
│   ├── figs/ → class_balance.png, tweet_length.png, confusion matrices
│   ├── metrics/ → baseline.csv, distilbert.csv
│   └── checkpoints/ → saved model weights
└── notebooks/ → Jupyter/Colab implementation
```

## Summary
- Completed data preprocessing, EDA, and baseline vs transformer comparison.
- DistilBERT achieved higher accuracy and F1 compared to the baseline.
- Next step: train for more epochs and add new transformer models like RoBERTa.