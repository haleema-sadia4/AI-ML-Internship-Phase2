# AI/ML Internship Phase 2 — DevelopersHub

Advanced AI/ML internship tasks covering transformers, ML pipelines, 
RAG chatbots, and LLM applications.

---

## Task 1 — News Topic Classifier Using BERT ✅

**Objective:** Fine-tune BERT to classify news headlines into 4 categories.

**Dataset:** AG News (120,000 examples via Hugging Face)

**Approach:**
- Tokenized text using bert-base-uncased
- Fine-tuned BERT using Hugging Face Trainer API
- Evaluated with Accuracy and F1-score
- Deployed live demo using Gradio

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | 90.2% |
| Macro F1 | 0.90 |
| Best Category | Sports (F1: 0.97) |

**Tools:** Hugging Face Transformers, Gradio, Google Colab (T4 GPU)

---
## Task 2 — End-to-End ML Pipeline for Customer Churn ✅

**Objective:** Build a production-ready ML pipeline to predict 
customer churn for a telecom company.

**Dataset:** IBM Telco Customer Churn (7,032 customers, 19 features)

**Approach:**
- Fixed hidden missing values in TotalCharges column
- Built ColumnTransformer handling numerical + categorical columns
- Chained preprocessing + model into one sklearn Pipeline
- Compared Logistic Regression vs Random Forest
- Used GridSearchCV (40 runs) to find optimal settings
- Exported final pipeline using joblib

**Results:**

| Model | Accuracy | Churned F1 |
|-------|----------|------------|
| Logistic Regression | 80% | 0.61 |
| Random Forest | 79% | 0.55 |
| Best (GridSearch) | 80% | 0.61 |

**Tools:** Scikit-learn Pipeline, GridSearchCV, joblib, Pandas
