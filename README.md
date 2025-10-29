# 🧠 Multi-Class Classification of Disaster-Related Tweets Using DistilBERT for Real-Time Emergency Response  

### 📅 Intermediate Update 1  

---

## 🧩 Work Completed  
- Loaded the Kaggle dataset: *Natural Language Processing with Disaster Tweets*  
- Cleaned missing values in the **‘keyword’** and **‘location’** columns  
- Performed **Exploratory Data Analysis (EDA)** with visualizations  
- Generated plots for:
  - Class distribution  
  - Tweet length histogram  
- Implemented two models:
  1. **TF-IDF + Logistic Regression** (Baseline Model)  
  2. **DistilBERT** (Fine-tuned Transformer, 2 Epochs)  
- Collected evaluation metrics and organized all outputs systematically  

---

## 📊 EDA Plots  
| File | Description |
|------|--------------|
| `class_balance.png` | Class distribution (0 = Non-Disaster, 1 = Disaster) |
| `tweet_length.png` | Tweet length histogram |

---

## 🤖 Model Performance  
| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| TF-IDF + Logistic Regression | 0.8070 | 0.7637 |
| DistilBERT (2 epochs) | 0.8437 | 0.8128 |

---

## 📈 Evaluation Figures  
| File | Description |
|------|--------------|
| `baseline_cm.png` | Confusion matrix for Logistic Regression |
| `distilbert_cm.png` | Confusion matrix for DistilBERT |

---

## 🗂️ Metrics Files  
| File | Description |
|------|--------------|
| `baseline.csv` | Accuracy and F1 Score of Logistic Regression |
| `distilbert.csv` | Accuracy and F1 Score of DistilBERT |

---

## 🧮 Training Curve  
| File | Description |
|------|--------------|
| `loss_curve_distilbert.png` | Training and Evaluation Loss over steps (DistilBERT) |

---

## 📁 Folder Layout  
ml-m1-disaster-tweets/
│
├── data/raw/ → train.csv, test.csv, sample_submission.csv
├── outputs/
│ ├── figs/ → class_balance.png, tweet_length.png, confusion matrices, loss_curve_distilbert.png
│ ├── metrics/ → baseline.csv, distilbert.csv
│ └── checkpoints/ → saved model weights
└── notebooks/ → Jupyter/Colab implementation



---

## ⚙️ Challenges Faced  
- Limited GPU runtime on Colab caused early stopping at 2 epochs for DistilBERT  
- Preprocessing complex tweets (hashtags, emojis, links) required additional cleaning  
- Managing large model files in GitHub (added `.gitignore` for checkpoints)  
- Transformer models required longer training time  

---

## 🚀 Next Steps  
- Extend DistilBERT training to 3–5 epochs for improved performance  
- Apply advanced text normalization (remove URLs, mentions, emojis)  
- Experiment with other transformer models like **RoBERTa** and **BERTweet**  
- Prepare the final project report and presentation slides for submission  

---

✅ **Status:**  
All requirements for *Intermediate Update 1* are completed successfully — including dataset preprocessing, EDA, baseline and transformer model implementation, metrics, and visual outputs.
