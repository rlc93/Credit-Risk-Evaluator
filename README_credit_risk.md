# Credit Risk Evaluator

A machine learning project that compares **Logistic Regression** and **Random Forest** classification models to predict loan risk using real-world lending data from 2019–2020.

---

## 📌 Overview

Credit risk assessment is a foundational challenge in consumer finance. This project trains and evaluates two supervised learning models on a dataset of loan applicants, using 2019 data for training and Q1 2020 data for testing — simulating how a model trained on historical data performs on future, unseen applicants.

---

## 📊 Results

| Model | Data | Accuracy |
|---|---|---|
| Logistic Regression | Unscaled | 55.87% |
| Random Forest | Unscaled | 63.06% |
| Logistic Regression | Scaled | 69.13% |
| **Random Forest** | **Scaled** | **79.64%** |

**Best model: Random Forest Classifier on scaled data — 79.64% accuracy**

Key finding: Feature scaling significantly improved both models, with the Random Forest classifier benefiting the most and outperforming Logistic Regression by ~10 percentage points on scaled data.

---

## 🗂️ Dataset

- **Training data:** `2019loans.csv` — loan records from 2019
- **Testing data:** `2020Q1loans.csv` — loan records from Q1 2020
- **Target variable:** `loan_status` (binary classification)

> Data sourced from LendingClub loan records. Categorical features were one-hot encoded, and missing dummy columns in the test set were imputed with zeros to ensure consistent feature alignment.

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, NumPy
- scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler)
- Jupyter Notebook

---

## 🔄 Methodology

1. **Data Loading** — Loaded separate train (2019) and test (2020 Q1) CSVs
2. **Preprocessing** — One-hot encoded categorical variables; aligned train/test columns
3. **Baseline Models** — Trained Logistic Regression and Random Forest on unscaled data
4. **Feature Scaling** — Applied `StandardScaler` to normalize feature distributions
5. **Scaled Models** — Retrained both models on scaled data and compared performance

---

## 📁 File Structure

```
Credit-Risk-Evaluator/
│
├── credit_risk_evaluator.ipynb   # Main analysis notebook
├── 2019loans.csv                 # Training dataset
├── 2020Q1loans.csv               # Testing dataset
└── README.md
```

---

## 💡 Key Takeaways

- Random Forest consistently outperformed Logistic Regression on this dataset
- Scaling is critical for Logistic Regression — unscaled performance was near-baseline
- Training on one year and testing on the next is a realistic evaluation strategy for time-sensitive financial data

---

## 👤 Author

**Randy Crystian Jr**  
Senior Analyst, Audience Development | Hilton  
[GitHub](https://github.com/rlc93)
