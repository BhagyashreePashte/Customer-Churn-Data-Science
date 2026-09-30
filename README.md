# 📊 Customer Churn Analysis & Prediction Using Python

<div align="center">

<a href="https://github.com/BhagyashreePashte/Customer-Churn-Data-Science">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&duration=3000&pause=1000&color=00C6FF&center=true&vCenter=true&width=700&lines=ChurnLens+AI;Customer+Retention+Intelligence;Customer+Churn+Prediction;Turning+Data+into+Insights" alt="Typing SVG" />
</a>

### 📊 Customer Retention Intelligence & Churn Prediction

**Turning Customer Data into Actionable Retention Insights**

</div>

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=35&duration=2500&pause=800&color=00C6FF&center=true&vCenter=true&width=800&lines=🧠+ChurnLens+AI;📊+Customer+Retention+Intelligence;🤖+AI-Powered+Churn+Prediction;💡+Data+Driven+Retention+Strategy" alt="ChurnLens AI" />

<br>

</div>



> **An end-to-end Data Science project for analyzing customer churn patterns, discovering actionable insights, performing statistical analysis, and developing predictive machine-learning models.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge\&logo=jupyter\&logoColor=white)

---

## 📌 Project Overview

Customer churn is a major business challenge in subscription-based industries. Identifying customers who are likely to leave can help organizations understand customer behavior, improve retention strategies, and make data-driven decisions.

This project develops an **end-to-end customer churn analytics and prediction pipeline** using the IBM Telco Customer Churn dataset.

The project progresses from:

**Raw Data → Data Cleaning → Exploratory Data Analysis → Data Visualization → Statistical Analysis → Machine Learning → Model Evaluation → Business Recommendations**

The project is being developed as part of the **YuvaIntern Virtual Data Science with Python Apprentice Internship**.

---

## 🎯 Project Objectives

The major objectives of this project are:

* Acquire and understand a real-world customer dataset.
* Perform systematic data cleaning and preprocessing.
* Explore customer demographics, services, contracts and billing behavior.
* Identify patterns associated with customer churn.
* Build informative and advanced data visualizations.
* Perform statistical hypothesis testing.
* Develop machine-learning models for churn prediction.
* Evaluate model performance using appropriate metrics.
* Identify important churn-related features.
* Translate analytical findings into practical business recommendations.

---

# 🧠 Project Workflow

```text
                    ┌─────────────────────┐
                    │    Raw Dataset     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Inspection     │
                    │ & Understanding     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Cleaning &     │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │ Analysis (EDA)      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Visualization &     │
                    │ Storytelling        │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Statistical        │
                    │ Hypothesis Testing  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │ Model Development   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Business Insights & │
                    │ Recommendations     │
                    └─────────────────────┘
```

---

# 📅 Project Development Roadmap

| Week   | Module                                      | Status      |
| ------ | ------------------------------------------- | ----------- |
| Week 1 | Data Acquisition, Cleaning & EDA            | ✅ Completed |
| Week 2 | Advanced Visualization & Storytelling       | 🔄 Planned  |
| Week 3 | Statistical Analysis & Hypothesis Testing   | 🔄 Planned  |
| Week 4 | Machine Learning & Model Evaluation         | 🔄 Planned  |
| Week 5 | Final Reporting & Strategic Recommendations | 🔄 Planned  |

---

# 📂 Dataset

### Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains information about telecom customers, their subscribed services, contracts, billing information and churn status.

### Dataset Characteristics

| Attribute              |   Value |
| ---------------------- | ------: |
| Original records       |   7,043 |
| Records after cleaning |   7,032 |
| Features               |      21 |
| Target variable        | `Churn` |
| Churn = Yes            |   1,869 |
| Churn = No             |   5,163 |
| Churn rate             |  26.58% |

### Major Variables

#### Customer Information

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`

#### Service Information

* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`

#### Account Information

* `tenure`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`

#### Financial Information

* `MonthlyCharges`
* `TotalCharges`

#### Target

* `Churn`

---

# 🧹 Data Cleaning

The raw dataset was inspected for:

* Missing values
* Duplicate records
* Incorrect data types
* Leading/trailing whitespace
* Invalid numerical values
* Inconsistent data representation

### Key preprocessing operations

```python
import pandas as pd

df_clean = df.copy()

# Remove unnecessary whitespace
object_columns = df_clean.select_dtypes(
    include="object"
).columns

for column in object_columns:
    df_clean[column] = df_clean[column].str.strip()

# Convert TotalCharges to numeric
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"],
    errors="coerce"
)

# Convert numerical columns
numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:
    df_clean[column] = pd.to_numeric(
        df_clean[column],
        errors="coerce"
    )

# Remove records with missing numerical values
df_clean = df_clean.dropna(
    subset=numeric_columns
)
```

The cleaned dataset is stored separately from the raw dataset to preserve data integrity.

---

# 📊 Week 1 — Exploratory Data Analysis

## Churn Distribution

After preprocessing:

* **73.42%** of customers did not churn.
* **26.58%** of customers churned.

This indicates that churn represents a significant portion of the customer population and warrants further investigation.

---

## 👥 Tenure Analysis

| Customer Group | Average Tenure |
| -------------- | -------------: |
| No Churn       |   37.65 months |
| Churn          |   17.98 months |

Customers who churned had considerably lower average tenure than customers who remained.

---

## 💰 Monthly Charges

| Customer Group | Average Monthly Charges |
| -------------- | ----------------------: |
| No Churn       |                   61.31 |
| Churn          |                   74.44 |

Churned customers had higher average monthly charges in this dataset.

---

## 📄 Contract Analysis

| Contract       | No Churn |      Churn |
| -------------- | -------: | ---------: |
| Month-to-month |   57.29% | **42.71%** |
| One year       |   88.72% |     11.28% |
| Two year       |   97.15% |      2.85% |

Contract type shows a substantial descriptive difference in churn patterns.

This relationship will be investigated further using formal statistical analysis.

---

# 🔗 Correlation Analysis

The numerical correlation matrix produced the following notable relationships:

| Variable Pair                  | Correlation |
| ------------------------------ | ----------: |
| Tenure ↔ TotalCharges          |    **0.83** |
| MonthlyCharges ↔ TotalCharges  |    **0.65** |
| Tenure ↔ MonthlyCharges        |        0.25 |
| SeniorCitizen ↔ MonthlyCharges |        0.22 |
| SeniorCitizen ↔ Tenure         |        0.02 |

The strongest relationship was observed between **tenure and TotalCharges (0.83)**.

> **Note:** Correlation represents statistical association and does not establish causation.

---

# 📈 Visualizations

The project includes multiple visualizations for understanding customer behavior.

### Current visualizations

* Churn distribution
* Tenure distribution
* Monthly charges distribution
* Total charges distribution
* Contract type vs churn
* Internet service vs churn
* Payment method vs churn
* Senior citizen vs churn
* Correlation heatmap
* Tenure vs monthly charges

All generated visualizations are stored in:

```text
visualizations/
```

---

# 🔬 Statistical Analysis

The Week 3 stage will investigate relationships identified during EDA using formal statistical methods.

Planned techniques include:

* Independent samples t-test
* Chi-square test of independence
* ANOVA where appropriate
* Confidence intervals
* p-value interpretation
* Effect-size interpretation

Example research question:

> **Is customer churn statistically associated with contract type?**

The statistical analysis will distinguish between descriptive patterns and statistically supported relationships.

---

# 🤖 Machine Learning

The Week 4 stage will develop predictive models for customer churn.

### Planned workflow

```text
Data
 ↓
Feature Selection
 ↓
Categorical Encoding
 ↓
Train/Test Split
 ↓
Feature Scaling
 ↓
Model Training
 ↓
Prediction
 ↓
Model Evaluation
```

### Candidate models

* Logistic Regression
* Decision Tree
* Random Forest

Additional models may be evaluated depending on project requirements.

---

# 📏 Model Evaluation

The models will be evaluated using multiple metrics rather than accuracy alone.

### Planned metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* ROC Curve

Special attention will be given to **recall**, because correctly identifying customers at risk of churn can be important in a customer-retention scenario.

---

# 💼 Business Insights

The analytical findings can help organizations investigate questions such as:

* Which customer segments have higher churn?
* Does contract duration relate to customer retention?
* Are higher monthly charges associated with churn?
* Does customer tenure influence retention?
* Which services are associated with different churn patterns?
* Which customer characteristics are most useful for prediction?

The final stage of the project will translate statistically supported and model-supported findings into practical recommendations.

---

# ⚠️ Limitations

The project has several limitations:

1. The dataset represents a specific telecom customer population.
2. EDA identifies associations but does not prove causation.
3. Churn classes are imbalanced.
4. Historical customer behavior may not represent future behavior.
5. Machine-learning performance depends on preprocessing, feature selection and model assumptions.
6. Business recommendations should be validated against current organizational data before implementation.

---

# 🚀 Future Enhancements

Potential future improvements include:

* Hyperparameter optimization
* Cross-validation
* Feature importance analysis
* SHAP-based model explainability
* Customer segmentation
* Churn probability dashboard
* Streamlit deployment
* Interactive Power BI dashboard
* Automated prediction pipeline
* Model monitoring

---

# 🛠️ Technology Stack

| Category          | Technologies              |
| ----------------- | ------------------------- |
| Programming       | Python                    |
| Data Manipulation | Pandas, NumPy             |
| Visualization     | Matplotlib, Seaborn       |
| Statistics        | SciPy, Statsmodels        |
| Machine Learning  | Scikit-learn              |
| Development       | Jupyter Notebook, VS Code |
| Version Control   | Git, GitHub               |
| Documentation     | Microsoft Word, Markdown  |

---

# 📁 Repository Structure

```text
Customer-Churn-Data-Science/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   │
│   └── processed/
│       └── Telco-Customer-Churn-Cleaned.csv
│
├── notebooks/
│   ├── Week1_EDA.ipynb
│   ├── Week2_Visualization.ipynb
│   ├── Week3_Hypothesis_Testing.ipynb
│   ├── Week4_ML_Model.ipynb
│   └── Week5_Final_Project.ipynb
│
├── visualizations/
│   ├── 01_churn_distribution.png
│   ├── 02_tenure_distribution.png
│   ├── 03_monthly_charges_distribution.png
│   ├── 04_total_charges_distribution.png
│   ├── 05_contract_vs_churn.png
│   ├── 06_internet_service_vs_churn.png
│   ├── 07_payment_method_vs_churn.png
│   ├── 08_senior_citizen_vs_churn.png
│   ├── 09_correlation_heatmap.png
│   └── 10_tenure_vs_monthly_charges.png
│
├── reports/
│   └── Week1_Customer_Churn_EDA_Report.docx
│
├── src/
│   ├── data_processing.py
│   ├── visualization.py
│   ├── statistical_analysis.py
│   └── model_training.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Data-Science.git
```

Navigate into the project:

```bash
cd Customer-Churn-Data-Science
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# ▶️ How to Use

### 1. Open the Week 1 notebook

```text
notebooks/Week1_EDA.ipynb
```

### 2. Run the cells sequentially.

### 3. Review generated visualizations.

### 4. Continue with the subsequent weekly notebooks as they are completed.

---

# 📚 Learning Outcomes

Through this project, the following practical Data Science skills are being developed:

* Data acquisition
* Data cleaning
* Data preprocessing
* Exploratory Data Analysis
* Statistical reasoning
* Data visualization
* Feature engineering
* Machine learning
* Model evaluation
* Business-oriented interpretation
* Technical documentation
* Git and GitHub workflow

---

# 👩‍💻 Author

## Bhagyashree Santosh Pashte

**B.E. Artificial Intelligence & Machine Learning**

Interested in:

* Artificial Intelligence
* Machine Learning
* Data Science
* Python
* Data Analytics
* Intelligent Systems

---
