# Multi-Class Classification of Disaster-Related Tweets Using DistilBERT for Real-Time Emergency Response

This project aims to automatically classify tweets as disaster-related or not. During natural disasters, thousands of tweets appear every minute, and manually checking them is impossible. Using machine learning and transformer-based models like DistilBERT, we can quickly filter disaster information and support real-time emergency response systems.

The dataset used is the Kaggle Disaster Tweets dataset, which contains around 10k tweets labeled as “disaster” (1) or “non-disaster” (0).
The goal of this project is to build ML and NLP models, compare their performance, and understand how different techniques improve tweet classification.

---

# **Intermediate Update 1 — Baseline Models & Initial DistilBERT Training**

## **Work Completed**

* Loaded Kaggle dataset: *Natural Language Processing with Disaster Tweets*
* Cleaned missing values in `keyword` and `location`
* Performed full EDA on tweet distribution, class imbalance, and tweet-length patterns
* Implemented two classification models:

  * **TF-IDF + Logistic Regression** (baseline)
  * **DistilBERT (fine-tuned, 2 epochs)**
* Generated evaluation metrics, confusion matrix, and training loss curves
* Organized outputs into clear GitHub project structure

---

## **EDA Plots**

| File                | Description                                     |
| ------------------- | ----------------------------------------------- |
| `class_balance.png` | Distribution of disaster vs non-disaster tweets |
| `tweet_length.png`  | Histogram of tweet lengths                      |

---

## **Model Performance (Update 1)**

| Model                        | Accuracy   | F1 Score   |
| ---------------------------- | ---------- | ---------- |
| TF-IDF + Logistic Regression | **0.8070** | **0.7637** |
| DistilBERT (2 epochs)        | **0.8437** | **0.8128** |

---

## **Evaluation Figures**

| File                        | Description                             |
| --------------------------- | --------------------------------------- |
| `baseline_cm.png`           | Confusion matrix of baseline model      |
| `loss_curve_distilbert.png` | Training/eval loss curve for DistilBERT |

---

## **Metrics Files**

| File             | Purpose                           |
| ---------------- | --------------------------------- |
| `baseline.csv`   | Logistic Regression accuracy & F1 |
| `distilbert.csv` | DistilBERT accuracy & F1          |

---

#  **Intermediate Update 2 — Multi-Task DistilBERT (Main + Auxiliary Task)**

## **Goal**

Enhance classification performance using a **Multi-Task Learning (MTL)** setup:

* **Main task:** Disaster (0/1)
* **Auxiliary task:** Tweet length bucket (0/1/2)

This approach helps the model learn structural patterns in text, improving generalization.

---

## **Work Completed**

* Added auxiliary target: tweet-length bucket (short / medium / long)
* Implemented **DistilBERT MTL model** with:

  * Shared transformer encoder
  * Main classification head (2 classes)
  * Auxiliary head (3 classes)
* Trained MTL model for **3 epochs**
* Printed epoch-wise accuracy & F1 metrics
* Saved all figures and metrics to GitHub
* Added MTL training curve under `outputs/figs/`

---

##  **Update 2 Training Curve**

| File                         | Description                                   |
| ---------------------------- | --------------------------------------------- |
| `loss_curve_mtl_update2.png` | Training loss curve for multi-task DistilBERT |

---

##  **MTL Model Evaluation (Update 2)**

| Epoch | Train Loss | Val Accuracy | Val F1 |
| ----- | ---------- | ------------ | ------ |
| 1     | 0.7405     | 0.8351       | 0.8064 |
| 2     | 0.4452     | 0.8266       | 0.8035 |
| 3     | 0.3049     | 0.8378       | 0.7950 |

---

## **Update 2 Metrics**

| File                         | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| `distilbert_mtl_update2.csv` | Stores all epoch-wise metrics for MTL model |

---

#  **Folder Layout (Final Combined Project)**

```
ML-disaster-tweets/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
│
├── outputs/
│   ├── figs/
│   │   ├── class_balance.png
│   │   ├── tweet_length.png
│   │   ├── baseline_cm.png
│   │   ├── loss_curve_distilbert.png
│   │   ├── loss_curve_mtl_update2.png
│   │   ├── label_distribution_update2.png
│   │
│   └── metrics/
│       ├── baseline.csv
│       ├── distilbert.csv
│       └── distilbert_mtl_update2.csv
│
├── Term_Project_Harsha_Update1.ipynb
├── Term_Project_Harsha_Update2.ipynb
├── README.md
└── .gitignore
```

---

# **Challenges Faced**

* Limited Colab GPU time (training longer epochs restricted)
* Tweets contained URLs, hashtags & emojis requiring extra preprocessing
* Large transformer checkpoints excluded via `.gitignore`
* Multi-task training and metrics extraction needed additional debugging

---

# **Status**

All deliverables for **Intermediate Update 1** and **Intermediate Update 2** are complete.
Dataset, EDA, baseline model, DistilBERT, multi-task model, metrics, and all visualizations are organized and pushed to GitHub.
The notebooks can be downloaded and run directly in Google Colab.

---
