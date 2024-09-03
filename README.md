# Loan Approval Prediction Project

This project aims to predict loan approval using various machine learning algorithms. It includes data preprocessing, exploratory data analysis (EDA), and the implementation of different classification models.

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)

## Introduction

This project analyzes a loan dataset to predict whether a loan application will be approved or not. It uses various machine learning algorithms such as Random Forest, Decision Trees, K-Nearest Neighbors, and Naive Bayes classifiers to make predictions.

## Dependencies

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

1. The dataset is loaded from a CSV file using pandas.
2. Initial data exploration is performed using `df.head()` and `df.info()`.
3. Missing values are identified using `df.isnull().sum()`.
4. Logarithmic transformation is applied to 'LoanAmount' and 'TotalIncome' to handle skewness.
5. Missing values are imputed using mode for categorical variables and mean for numerical variables.

## Exploratory Data Analysis

The project includes several visualizations to understand the distribution of various features:

1. Histogram of log-transformed loan amount
2. Histogram of log-transformed total income
3. Count plots for categorical variables:
   - Gender
   - Marital Status
   - Number of Dependents
   - Loan Amount
   - Credit History

These visualizations help in understanding the distribution of different features and their potential impact on loan approval.

## Feature Engineering

1. A new feature 'TotalIncome' is created by summing 'ApplicantIncome' and 'CoapplicantIncome'.
2. Log transformation is applied to 'TotalIncome' to handle skewness.

## Model Training and Evaluation

The dataset is split into training and test sets using a 80-20 ratio. The following steps are performed:

1. Label encoding is applied to categorical variables.
2. Feature scaling is done using StandardScaler.
3. Four different classifiers are trained and evaluated:
   - Random Forest Classifier
   - Decision Tree Classifier
   - K-Nearest Neighbors Classifier
   - Gaussian Naive Bayes Classifier

## Results

The accuracy of each classifier is printed:

- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Gaussian Naive Bayes Classifier

(The actual accuracy scores will be displayed when you run the code)

## Visualizations

The project includes several visualizations that are generated using seaborn and matplotlib. These visualizations are not included in this README but will be displayed when you run the code. They include:

1. Histogram of log-transformed loan amount
2. Histogram of log-transformed total income
3. Count plots for Gender, Marital Status, Dependents, Loan Amount, and Credit History

These visualizations provide insights into the distribution of various features in the dataset and can help in understanding patterns that might influence loan approval.

To view these visualizations, run the code and refer to the output plots.

## Conclusion

This project demonstrates the process of building a loan approval prediction model, from data preprocessing and exploratory data analysis to model training and evaluation. By comparing different classifiers, we can identify which model performs best for this particular dataset.

Feel free to explore the code, modify the models, or add new features to improve the prediction accuracy!
