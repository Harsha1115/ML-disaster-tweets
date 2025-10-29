##  Multi-Class Classification of Disaster-Related Tweets Using DistilBERT for Real-Time Emergency Response

###  Intermediate Update 1 — Model Implementation & Initial Training

---

### ✅ Work Completed

* Loaded the Kaggle dataset: *Natural Language Processing with Disaster Tweets*
* Cleaned missing values in **keyword** and **location** columns
* Performed Exploratory Data Analysis (EDA) on tweet distribution and length
* Implemented two models:

  * **TF-IDF + Logistic Regression** → Baseline
  * **DistilBERT** (fine-tuned transformer, 2 epochs)
* Collected evaluation metrics, generated plots, and saved organized outputs

---

### 📊 EDA Plots

| Plot                | Description                                         |
| :------------------ | :-------------------------------------------------- |
| `class_balance.png` | Class distribution (0 = Non-Disaster, 1 = Disaster) |
| `tweet_length.png`  | Tweet-length histogram                              |

---

###  Model Performance

| Model                        | Accuracy | F1 Score |
| :--------------------------- | :------: | :------: |
| TF-IDF + Logistic Regression |  0.8070  |  0.7637  |
| DistilBERT (2 epochs)        |  0.8437  |  0.8128  |

---

###  Evaluation Figures

| File                             | Description                                          |
| :------------------------------- | :--------------------------------------------------- |
| `baseline_cm.png`                | Confusion matrix for baseline model                  |
| `loss_curve_distilbert.png`      | Training/Eval loss curve for DistilBERT              |

---

###  Metrics Files

| File             | Purpose                               |
| :--------------- | :------------------------------------ |
| `baseline.csv`   | Accuracy & F1 for Logistic Regression |
| `distilbert.csv` | Accuracy & F1 for DistilBERT model    |

---

###  Folder Layout

```
ml-m1-disaster-tweets/
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
│   │
│   ├── metrics/
│   │   ├── baseline.csv
│   │   └── distilbert.csv
│   │
│   └── checkpoints/            # saved model weights
│
└── README.md
```

---

###  Challenges Faced

* Limited Colab GPU time → DistilBERT training stopped at 2 epochs
* Tweets contained hashtags, links & emojis → needed extra pre-processing
* Large model files excluded from GitHub via `.gitignore`
* Transformer training was computationally expensive

---

### 🔜 Next Steps

* Extend DistilBERT training to 3–5 epochs for better accuracy
* Apply advanced text normalization (remove URLs, mentions, emojis)
* Experiment with other transformer models (RoBERTa, BERTweet)
* Prepare final report and presentation slides

---

### 📉 Training Curve

* `loss_curve_distilbert.png` → Shows training vs evaluation loss trend for DistilBERT

---

✅ **Status:** All deliverables for Intermediate Update 1 are completed and pushed to GitHub — dataset, EDA, baseline model, transformer model, metrics, and visualizations ready for review.

---
