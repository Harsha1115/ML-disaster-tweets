#  **Multi-Class Classification of Disaster-Related Tweets Using DistilBERT**

This project focuses on automatically identifying disaster-related tweets using machine learning and transformer-based NLP techniques. During natural disasters, thousands of tweets appear instantly, and identifying relevant information becomes essential for emergency response. By fine-tuning models like DistilBERT and experimenting with multi-task learning, this project demonstrates how tweet classification can be improved for real-time use.

The dataset used is the **Kaggle Disaster Tweets dataset**, containing real tweets labeled 0 as non-disaster and 1 as disaster.
The project was developed in two stages:

* **Intermediate Update 1:** EDA, baseline models, and single-task DistilBERT
* **Intermediate Update 2:** Multi-task DistilBERT with auxiliary tweet-length prediction

All code, outputs, figures, and metrics are included in this repository.

---

# **Intermediate Update 1 — Baseline & DistilBERT Fine-Tuning**

Update 1 focused on building the foundation of the project.
We explored the dataset, cleaned text features, and trained the first two models.

##  **Work Completed**

* Loaded and cleaned the Kaggle dataset
* Performed EDA (class imbalance, tweet length patterns, text distribution)
* Implemented:

  * **TF-IDF + Logistic Regression** (baseline)
  * **Single-task DistilBERT** (fine-tuned for 2 epochs)
* Generated classification reports, confusion matrix, and loss curves
* Saved all outputs under `outputs/figs/` and metrics under `outputs/metrics/`

---

#  **EDA Figures (Update 1)**

| File                | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `class_balance.png` | Shows distribution of 0 vs 1 classes (non-disaster vs disaster) |
| `tweet_length.png`  | Tweet-length histogram to understand text size patterns         |

---

#  **Model Performance (Update 1)**

| Model                        | Accuracy | F1-Score |
| ---------------------------- | -------- | -------- |
| TF-IDF + Logistic Regression | 0.8070   | 0.7637   |
| DistilBERT (2 epochs)        | 0.8437   | 0.8128   |

---

# **Evaluation Figures (Update 1)**

| File                        | Description                         |
| --------------------------- | ----------------------------------- |
| `baseline_cm.png`           | Confusion matrix for baseline model |
| `loss_curve_distilbert.png` | Loss curve for DistilBERT training  |

---

#  **Metrics Files (Update 1)**

| File             | Purpose                        |
| ---------------- | ------------------------------ |
| `baseline.csv`   | Accuracy & F1 for TF-IDF model |
| `distilbert.csv` | Accuracy & F1 for DistilBERT   |

---

#  **Intermediate Update 2 — Multi-Task DistilBERT (Main + Auxiliary Task)**

Update 2 extended the project by upgrading the model from single-task to multi-task learning.
Multi-task learning helps the model learn deeper representations by adding an additional objective.

##  **What Changed in Update 2**

* Introduced **auxiliary task:** Tweet-length bucket prediction

  * (Short / Medium / Long → 3 classes)
* Built custom **multi-head architecture**:

  * Shared DistilBERT encoder
  * Main classification head (2 classes)
  * Aux head (3 classes)
* Joint training using combined loss:

  ```
  total_loss = main_loss + 0.5 * auxiliary_loss
  ```
* Trained the model for 3 epochs
* Logged training loss + validation accuracy + validation F1
* Saved updated figures, metrics, and summary tables

---

# **Update 2: Training Curve**

| File                         | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `loss_curve_mtl_update2.png` | Multi-task training loss curve across 3 epochs |

---

# **Update 2 Evaluation Results**

| Epoch | Train Loss | Val Accuracy | Val F1 |
| ----- | ---------- | ------------ | ------ |
| 1     | 0.7405     | 0.8351       | 0.8064 |
| 2     | 0.4452     | 0.8266       | 0.8035 |
| 3     | 0.3049     | 0.8378       | 0.7950 |

The multi-task model showed smoother training and more stable performance across epochs.

---

#  **Metrics Files Update 2**

| File                         | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| `distilbert_mtl_update2.csv` | All epoch-wise metrics for multi-task DistilBERT |

---

#  **Other Update 2 Figures**

| File                             | Description                             |
| -------------------------------- | --------------------------------------- |
| `label_distribution_update2.png` | Distribution of auxiliary label buckets |
| `loss_curve_mtl_update2.png`     | Multi-task DistilBERT loss curve        |

---

#  **Final Repository Structure**

```
ML-disaster-tweets/
│
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

This structure follows the same organization across both updates, making evaluation easy and professional.

---

#  **Final Note (Same Style as Your Old README)**

**Status: All deliverables for Intermediate Update 1 and Intermediate Update 2 have been completed and pushed to GitHub — including the dataset, EDA outputs, baseline model, DistilBERT single-task model, multi-task DistilBERT model, metrics, and all visualizations. The project is fully organized and ready for final evaluation.**

---
