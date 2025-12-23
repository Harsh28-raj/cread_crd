# 💳 Credit Card Fraud Detection System  
### End-to-End Machine Learning Project

An industry-aligned machine learning project that detects fraudulent credit card transactions using advanced feature engineering, ensemble learning, and real-time deployment with Streamlit.

---

## 📌 Project Overview

With the rapid growth of digital payments, credit card fraud has become a serious challenge for financial institutions. Fraud cases are rare, highly imbalanced, and costly if missed.

This project builds a **complete end-to-end machine learning pipeline** to detect fraudulent transactions with a **strong focus on fraud recall**, which is critical in real-world fraud detection systems.

The project covers:
- Data analysis & feature engineering  
- Model experimentation & evaluation  
- Handling imbalanced data  
- Model serialization  
- Real-time deployment using Streamlit  

---

## 🎯 Problem Statement

**Objective:**  
To predict whether a credit card transaction is **fraudulent or normal** using transaction, temporal, and geographical features.

**Key Challenges:**
- Fraud cases are extremely rare
- Highly imbalanced dataset
- Accuracy alone is misleading
- False negatives (missed frauds) are very costly

**Business Priority:**  
> Maximize **fraud recall**, even at the cost of slightly lower precision.

---

## 📊 Dataset Information

- **Original Dataset Size:** `1,048,576` transactions  
- **Modeling Dataset:** Cleaned & stratified subset used for efficient training  
- **Target Variable:** `is_fraud`

### Class Distribution

| Class  | Count | Percentage |
|------|------|------------|
| Normal | ~99.43% | Majority |
| Fraud | ~0.57% | Minority |

> This extreme imbalance strongly influenced model choice and evaluation metrics.

---

## 🧾 Features Used

| Feature | Description |
|------|------------|
| `amt` | Transaction amount |
| `category` | Merchant category |
| `gender` | Customer gender |
| `state`, `zip` | Customer location |
| `lat`, `long` | Customer coordinates |
| `merch_lat`, `merch_long` | Merchant coordinates |
| `city_pop` | City population |
| `trans_date_trans_time` | Transaction timestamp |
| `is_fraud` | Target label |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to:
- Understand class imbalance
- Identify fraud-related behavioral patterns
- Guide feature engineering decisions

### Key Insights:
- Fraud transactions are extremely rare
- Fraud often occurs at **odd hours**
- High transaction amounts show abnormal behavior
- Large distance between customer & merchant increases fraud likelihood

---

## 🛠️ Feature Engineering

Key transformations:
- **Time Features:** Hour of day, day of week
- **Geographical Features:** Customer–merchant distance
- **Amount Handling:** Log transformation for skewed values
- **Categorical Encoding:** One-hot encoding with `handle_unknown="ignore"`
- **Feature Selection:** Noise reduction to avoid overfitting

---

## 🤖 Models Evaluated

| Model | Observation |
|------|------------|
| Logistic Regression | Poor fraud recall |
| Decision Tree | Overfitting risk |
| Random Forest | Strong performance but weaker generalization |
| **XGBoost (Final)** | Best recall & generalization |

### ✅ Why XGBoost?
- Handles imbalanced data effectively  
- Learns complex feature interactions  
- Strong real-world generalization  

---

## 📈 Model Evaluation

**Metrics Used:**
- Fraud Recall (most important)
- Precision
- F1-Score
- ROC-AUC

**Final XGBoost Results (Approximate):**
- Accuracy: ~98%
- Fraud Recall: ~96%
- ROC-AUC: Highest among all tested models

> The model prioritizes **catching fraud**, not just accuracy.

---

## 💾 Model Serialization

The trained model was saved using `joblib`:







cread_crd/
│
├── app.py                 
├── fraud_xgb_model.pkl     
├── requirements.txt       
├── README.md               
├── notebooks/              
├── docs/                   
└── .gitignore

```python
joblib.dump(xgb_model, "fraud_xgb_model.pkl")
