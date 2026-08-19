# Classification Model - Insurance Cross-Selling Prediction

---

## Overview

This project consists of a machine learning application developed in Python to predict whether existing health insurance customers are likely to purchase an additional vehicle insurance policy.

The project combines data preprocessing, exploratory data analysis, class imbalance handling and binary classification using Logistic Regression.

The developed solution allows users to:

- explore the dataset and identify relevant patterns;
- analyse the relationship between customer characteristics and purchase behaviour;
- handle class imbalance using different techniques;
- train and evaluate a Logistic Regression classifier;
- compare different decision thresholds;
- compare class-weight balancing and oversampling strategies;
- identify the most appropriate predictive model.

The project was developed following a predefined specification, with particular attention to data exploration, model evaluation and business-oriented interpretation of the results.

---

## Technologies

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn

---

## Project Requirements

### Insurance Cross-Selling

The objective of the project is to develop a predictive model capable of identifying customers who are more likely to accept a vehicle insurance cross-selling offer.

The project includes the following tasks:

- Explore the dataset.
- Analyse the distribution of the target variable.
- Investigate the relationships between customer characteristics and the target variable.
- Handle class imbalance using appropriate techniques.
- Develop a binary classification model.
- Evaluate the model using appropriate performance metrics.
- Compare different modelling strategies and identify the most suitable solution.

---

## Dataset

The dataset contains demographic, behavioural and insurance-related information about existing customers.

The main features include:

- Customer gender
- Customer age
- Driving licence ownership
- Region of residence
- Previous vehicle insurance status
- Vehicle age
- Previous vehicle damage
- Annual insurance premium
- Policy sales channel
- Customer tenure (Vintage)

Target variable:

- **Response = 1** → Customer accepted the cross-selling offer.
- **Response = 0** → Customer rejected the offer.

The dataset used in this project was provided by ProfessionAI and is available at: 

[Cross Selling dataset](https://proai-datasets.s3.eu-west-3.amazonaws.com/insurance_cross_sell.csv)

---

## Methodology

The project includes the following stages:

- Data preprocessing through categorical feature encoding.
- Feature standardisation using StandardScaler.
- Exploratory Data Analysis (EDA).
- Univariate analysis.
- Multivariate analysis.
- Logistic Regression model development.
- Class imbalance handling through:
  - `class_weight="balanced"`;
  - Random Oversampling.
- Comparison of different decision thresholds.
- Model evaluation using:
  - Classification Report;
  - Confusion Matrix;
  - ROC Curve;
  - Area Under the ROC Curve (AUC).

---

## Project Structure

```text
1.Classification_Model_Cross_Selling/
│
├── README.md
├── .gitignore
└── Classification_model_cross_selling.ipynb
```

---

## Main Findings

The exploratory analysis shows that:

- **Previously_Insured** is the feature most strongly associated with customer response.
- Customers owning older vehicles are more likely to accept the cross-selling offer.
- Previous vehicle damage is associated with a substantially higher acceptance rate.
- The apparent effect of customer age largely disappears after conditioning on previous insurance status.

The modelling phase shows that:

- Logistic Regression with `class_weight="balanced"` provides the best overall performance.
- The model achieves an AUC of approximately **0.83** while maintaining high recall on the minority class.
- Increasing the decision threshold substantially reduces recall without providing a proportional improvement in precision.
- Lowering the decision threshold increases false positives while providing only marginal gains in recall.
- Random Oversampling performs worse than class-weight balancing for this dataset.

Overall, the Logistic Regression model using `class_weight="balanced"` and the default decision threshold of **0.5** provides the best trade-off between predictive performance and robustness.

---

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Run the project:

```bash
python classification_model_cross_selling.py
```

The script automatically performs:

- data preprocessing;
- exploratory data analysis;
- model development;
- model evaluation;
- comparison of different class imbalance strategies;
- presentation of the final results.
