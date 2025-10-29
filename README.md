# ML Milestone 1 - Disaster Tweets Classification (Weeks 2-3)

**Author:** Harsha Reddy Kalchuri  
**Project:** Model Implementation & Initial Training  

## Dataset
Kaggle - *Natural Language Processing with Disaster Tweets*  

Files:
- train.csv, test.csv, sample_submission.csv

## Models Implemented
| Model | Accuracy | F1 Score |
|--------|-----------|-----------|
| TF-IDF + Logistic Regression | 0.8070 | 0.7637 |
| DistilBERT (2 epochs) | 0.8437 | 0.8128 |

## Outputs
Plots and metrics saved under:
outputs/
├─ figs/ → EDA plots + confusion matrix  
├─ metrics/ → baseline.csv, distilbert.csv  
└─ checkpoints/ → ignored (large weights)

## Next Steps
- Fine-tune DistilBERT (3–5 epochs)
- Try RoBERTa & BERTweet
- Add text preprocessing (URLs, hashtags)
- Plot training loss & accuracy for final report
