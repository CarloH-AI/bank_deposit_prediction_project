# Bank-Deposit Prediction Project 💰🔮

A compact, end-to-end **binary-classification** workflow that predicts whether a prospective client of a Portuguese bank will subscribe to a term deposit.  
The notebook shows the full path from raw data to a tuned model—ideal as a template for structured-tabular ML work.

---

## Folder layout

```

bank\_deposit\_prediction\_project/
├── dataset/
│   └── bank-full.csv           # 45 211 rows × 17 features (UCI Bank Marketing)
├── Bank\_Deposit\_Prediction\_Project.ipynb
└── Project Report.pdf          # short paper-style write-up

````
*The raw CSV is the “**bank-full**” version of the Bank-Marketing dataset*

---

## Quick start

```bash
git clone https://github.com/CarloH-AI/bank_deposit_prediction_project.git
cd bank_deposit_prediction_project

# optional: clean Python env
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt      # or use the minimal list below

# open the notebook
jupyter lab Bank_Deposit_Prediction_Project.ipynb
````

**Minimal requirements**

```
python >= 3.10
pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost jupyter
```

---

## Pipeline in a nutshell

| Step                   | Key actions                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **1 – EDA**            | Class balance check (≈ 11 % positive), feature inspection, correlation heat-map.                                           |
| **2 – Pre-processing** | One-hot encode categoricals, scale continuous vars, handle class imbalance with **SMOTE**.                                 |
| **3 – Model sweep**    | Benchmark **Logistic Regression**, **Random Forest**, and **XGBoost**; 5-fold cross-validation over hyper-parameter grids. |
| **4 – Evaluation**     | Metrics: ROC-AUC, F1, Recall; plots: ROC curve and Precision-Recall.                                                       |
| **5 – Findings**       | Tuned **XGBoost** reaches ≈ 0.92 ROC-AUC on the hold-out set; top drivers are *duration*, *poutcome*, and *contact type*.  |
| **6 – Next steps**     | SHAP-based interpretability, cost-sensitive thresholds, pipeline export with `skops`.                                      |

Dataset reference: UCI Bank Marketing (45 211 records, 17 features, target **y**) ([archive.ics.uci.edu][1])

---

## Results at a glance

* **Best model** – XGBoost (learning-rate 0.05, 300 rounds, `max_depth=5`)
* **Hold-out ROC-AUC** – \~0.92
* **Lift\@10 %** – \~2.8× over random (see notebook section “Business lift”)

> The model improves recall on the minority “yes” class by applying SMOTE and threshold tuning, while maintaining precision suitable for targeted follow-ups.

[1]: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing "UCI Machine Learning Repository"
