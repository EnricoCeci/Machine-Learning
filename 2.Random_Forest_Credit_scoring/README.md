# Credit Scoring Model

---

## Overview

This project consists of a machine learning application developed in Python to predict customers' creditworthiness and support credit card approval decisions.

The project combines data cleaning, exploratory data analysis, categorical feature encoding, class imbalance handling and binary classification using Decision Tree and Random Forest models.

The developed solution allows users to:

- explore the dataset and analyse the distribution of customer creditworthiness;
- identify and handle missing or inconsistent data;
- investigate the relationship between income, occupation and socio-demographic characteristics and creditworthiness;
- preprocess categorical and numerical features for machine learning;
- handle class imbalance using class-weight balancing;
- train and evaluate an interpretable Decision Tree classifier;
- train and evaluate a Random Forest classifier as a more robust predictive reference;
- compare the trade-off between recall and false positives;
- interpret individual Decision Tree classification paths to support explanations of credit decisions.

The project was developed following a predefined specification, with particular attention to model interpretability, class imbalance and the business implications of classification errors.

---

## Technologies

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Graphviz

---

## Project Requirements

### Creditworthiness Prediction

The objective of the project is to develop a predictive model capable of estimating whether a customer has high creditworthiness and can therefore support the bank in assessing credit card applications.

The project includes the following tasks:

- Explore and clean the dataset.
- Analyse the distribution of the target variable.
- Investigate the relationships between customer characteristics and creditworthiness.
- Preprocess categorical and numerical variables.
- Handle the strong imbalance between target classes.
- Develop binary classification models.
- Evaluate model performance using appropriate classification metrics.
- Compare an interpretable Decision Tree with a Random Forest model.
- Provide interpretable information that can help explain customer classifications.

---

## Dataset

The dataset contains demographic, income, employment, housing and contact information about customers.

After data cleaning, the dataset contains **338,426 observations**.

The main features include:

- Customer gender
- Car ownership
- Property ownership
- Number of children
- Annual income
- Income type
- Education level
- Family status
- Housing type
- Age expressed as days from birth
- Employment history expressed in days
- Mobile phone availability
- Work phone availability
- Phone availability
- Email availability
- Occupation type
- Number of family members

Target variable:

- **TARGET = 1** → Customer has high creditworthiness and consistently repays instalments.
- **TARGET = 0** → Customer is not classified as highly creditworthy.

The target variable is strongly imbalanced:

- approximately **91.2%** of customers belong to class 0;
- approximately **8.8%** belong to class 1.

---

## Methodology

The project includes the following stages:

- Dataset inspection and data quality analysis.
- Removal of one observation affected by multiple missing values.
- Replacement of missing `OCCUPATION_TYPE` values with `Unknown`.
- Binary encoding of car and property ownership variables.
- Exploratory Data Analysis (EDA).
- Bivariate analysis using:
  - contingency tables for categorical variables;
  - boxplots for quantitative variables.
- Multivariate analysis focused on annual income in combination with:
  - occupation type;
  - number of children;
  - family status.
- Stratified train-test split to preserve target class proportions.
- Feature preprocessing using `ColumnTransformer`:
  - One-Hot Encoding for nominal categorical variables;
  - Ordinal Encoding for education level;
  - passthrough of numerical and binary variables.
- Removal of the customer `ID` from the predictive features.
- Class imbalance handling using `class_weight="balanced"`.
- Decision Tree development and comparison of tree depths.
- Random Forest development and comparison of different numbers of estimators.
- Model evaluation using:
  - Accuracy;
  - Precision;
  - Recall;
  - F1-score;
  - Classification Report;
  - Confusion Matrix.
- Decision Tree visualisation through Graphviz to support model interpretability.

---

## Project Structure

```text
credit-scoring/
│
├── README.md
└── Credit_Scoring_Model_EN.ipynb
```

---

## Main Findings

The exploratory analysis shows that:

- Annual income alone has limited discriminatory power, since the distributions of reliable and non-reliable customers strongly overlap.
- Income type does not clearly distinguish between the two target classes.
- Occupation type shows some differences across categories, although its overall discriminatory power remains limited.
- The `Unknown` occupation category is associated with lower creditworthiness.
- The number of children shows only a weak and non-monotonic relationship with creditworthiness.
- Family status does not appear to have a substantial independent effect on the target.
- Multivariate analyses involving annual income, occupation type, number of children and family status do not reveal simple interactions or clear discriminatory thresholds.

Overall, the exploratory analysis suggests that creditworthiness cannot be explained by a single dominant feature and is instead associated with combinations of multiple customer characteristics.

The modelling phase shows that:

- A Decision Tree with `max_depth=4` and `class_weight="balanced"` provides almost identical performance on the training and test sets.
- The selected Decision Tree achieves approximately **0.948 test accuracy**.
- For the positive class, the Decision Tree achieves approximately **0.99 recall**, **0.63 precision** and an **F1-score of 0.77**.
- The very high recall allows the model to identify almost all customers classified as reliable, but this comes at the cost of a substantial number of false positives.
- Increasing the tree depth from 4 to 5 does not provide a meaningful improvement.
- Additional constraints through different `min_samples_leaf` values do not improve the Decision Tree.
- Random Forest performance stabilises at around **0.949 test accuracy**.
- A Random Forest with **500 trees** achieves approximately **0.745 recall** and an **F1-score of 0.719** on the positive class.
- Increasing the number of trees beyond 500 provides only marginal improvements.
- Compared with the Decision Tree, the Random Forest identifies fewer reliable customers but substantially reduces the number of false positives.

The final modelling choice therefore depends on the bank's business objective:

- prioritising **recall** if the goal is to identify as many potentially reliable customers as possible;
- prioritising a reduction in **false positives** if the goal is to reduce the risk of approving customers who may subsequently default.

---

## Model Interpretability

Interpretability is an important requirement of the project because customers whose credit card application is rejected may need to receive an explanation of the decision.

The Decision Tree supports this requirement by providing explicit decision paths based on customer characteristics.

For example, one path in the trained tree identifies a reliable customer through a combination of:

- annual income above approximately **$163,000**;
- age above approximately **42–43 years**;
- more than approximately **4 years of employment experience**.

These decision rules should not be interpreted as causal relationships.

The model does not establish that a specific level of income, age or employment history causes a customer to repay or default. Instead, it identifies statistical patterns observed among previous customers and uses those patterns to classify new applicants.

The Decision Tree can therefore support an explanation of **how the model reached a classification**, while avoiding claims about the actual causes of customer payment behaviour.

---

## How to Run

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn graphviz
```

Open the notebook in Jupyter Notebook, JupyterLab or Google Colab:

```text
Credit_Scoring_Model_EN.ipynb
```

Make sure that the `credit_scoring.csv` dataset is available and update the dataset path in the import cell if necessary.

Then run the notebook cells in order.

The notebook automatically performs:

- data loading and inspection;
- data cleaning;
- exploratory data analysis;
- categorical feature encoding;
- train-test splitting;
- Decision Tree training and evaluation;
- Decision Tree visualisation;
- Random Forest training and evaluation;
- comparison of the two modelling approaches;
- interpretation of the final results.
