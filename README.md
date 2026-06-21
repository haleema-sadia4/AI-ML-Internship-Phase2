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

## Task 3 — Multimodal Housing Price Prediction (CNN + Tabular) ✅

**Objective:** Predict house prices using both property images and 
structured tabular data (bedrooms, bathrooms, area).

**Dataset:** Houses Dataset (Ahmed & Moustafa) — 535 houses with 
frontal images + tabular features

**Approach:**
- Extracted visual features using pre-trained MobileNetV2 (transfer learning)
- Processed tabular features through dense layers
- Fused both feature sets via concatenation
- Identified model collapse issue (predicting near-constant values)
- Fixed by scaling target variable — improved MAE by 51%
- Evaluated using MAE and RMSE, visualized actual vs predicted

**Results:**

| Version | MAE | RMSE |
|---------|-----|------|
| Without target scaling | $432,844 | $558,585 |
| With target scaling | $213,224 | $305,968 |

**Key insight:** Model performs best on mid-range houses ($400K-$800K), 
tends to regress toward the mean for very cheap or very expensive 
properties — a common limitation with small datasets (428 training examples).

**Tools:** TensorFlow/Keras, MobileNetV2, Scikit-learn, Pandas

## Task 4 — Context-Aware Chatbot Using RAG ✅

**Objective:** Build a conversational chatbot that remembers context 
and retrieves answers from a custom knowledge document.

**Knowledge base:** Custom AI overview document (history, applications, 
ethics, machine learning, NLP, computer vision)

**Approach:**
- Split document into chunks using RecursiveCharacterTextSplitter
- Generated embeddings using Google's gemini-embedding-001 model
- Built a FAISS vector store for semantic similarity search
- Implemented conversational memory to handle follow-up questions
- Diagnosed and resolved free-tier API rate limiting by building a 
  lightweight custom retrieval function (reduced API calls per 
  question from ~5 to 2)
- Deployed as an interactive chatbot using Gradio

**Demonstrated capabilities:**

| Question | Tests |
|----------|-------|
| "What is AI?" | Basic RAG retrieval |
| "What are some applications of it?" | Pronoun resolution via memory ("it" = AI) |
| "Can you give an example from what you just mentioned?" | Multi-turn memory across 2+ exchanges |

**Key insight:** LangChain's default ConversationalRetrievalChain makes 
multiple embedding calls per question (question reformulation + 
retrieval), which exhausted free-tier API quota. Rebuilt a leaner 
manual RAG pipeline (1 retrieval + 1 generation call) that preserved 
full functionality while working within free-tier limits.

**Tools:** LangChain, Google Gemini API (gemini-2.5-flash), 
FAISS vector store, Gradio
