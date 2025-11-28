# 🚀 Multi-Class Classification of Disaster-Related Tweets Using DistilBERT


This project focuses on building a machine learning system that can automatically detect whether a tweet is related to a real disaster. Social media posts become extremely important during earthquakes, floods, fires, and other emergencies. Since thousands of tweets get posted every minute, manually filtering them is not possible.
The goal of this project is to use NLP and transformer models to classify disaster-related tweets accurately and efficiently.

---

#  **Overview of the Project**

We used the **Kaggle Disaster Tweets dataset**, which contains real tweets labeled as *disaster* or *non-disaster*.
The project was completed in two phases:

1. **Intermediate Update 1 → Baseline & Single-task DistilBERT**
2. **Intermediate Update 2 → Multi-task DistilBERT**

Both updates follow a clear workflow:
data preparation → EDA → modeling → evaluation → results → organized outputs.

---

#  **Intermediate Update 1 — Baseline & DistilBERT**

The first update focused on understanding the dataset and building the initial models.
After cleaning and inspecting the data, several visualizations were created to study class imbalance and tweet length patterns. These plots helped guide model selection and prepare the data for training.

A simple baseline model using **TF-IDF + Logistic Regression** was trained first. This model provided a starting point to compare all future improvements.
After that, a **DistilBERT transformer model** was fine-tuned on the dataset for two epochs. DistilBERT, being more sophisticated, learned contextual meaning from text and achieved higher accuracy and F1-score.

All graphs, confusion matrices, and metrics were saved into organized folders for easy review.

---

# 🟦 **Intermediate Update 2 — Multi-Task DistilBERT**

In Update 2, the project moved beyond single-task classification.
Instead of predicting only whether a tweet is disaster-related, a second auxiliary task was added: **predicting the tweet length category (short/medium/long)**.

A custom **multi-task DistilBERT architecture** was created:

* One shared DistilBERT encoder
* Two classifier heads:

  * Main head → disaster classification
  * Auxiliary head → tweet-length bucket prediction

Both tasks are trained together. The combined loss helps the model learn richer patterns and improves stability during training.
The multi-task model was trained for three epochs, and the training loss, validation accuracy, and F1-score were recorded and plotted.

All metrics were exported as CSV files and saved exactly like Update 1, following the same structure.

---

# 🟦 **Results Summary**

**Update 1:**

* Baseline model gave moderate performance
* DistilBERT significantly improved accuracy and F1
* Training was smooth and visually represented with clean graphs

**Update 2:**

* Multi-task DistilBERT showed stable loss reduction across epochs
* Validation accuracy remained consistent
* Multi-task learning helped the model better understand tweet structure
* All results were captured and saved to the repository

Both updates follow a clean, consistent workflow and show progressive improvements in model capability.

---

# 🟦 **Repository Structure**

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
└── README.md
```

The structure is identical for both updates, making the project easy to navigate and evaluate.

---

# 🟦 **Final Note**

This repository now contains the complete work for both Update 1 and Update 2, including all data, EDA, baseline and transformer models, multi-task training, metrics, and visualizations.
Everything is organized, verified, and ready for final evaluation.

---
