# Wine-Quality-Prediction-Using-ML

## **Overview**

This project builds machine learning models to predict the quality of red wine based on physicochemical tests.
We explore multiple ML algorithms, perform data preprocessing, visualize insights, and evaluate model performance to determine the best wine quality predictor.

## **Dataset**


Source: UCI Wine Quality Dataset

File Used: winequality-red.csv

Attributes:

11 input features (like acidity, alcohol, pH)

1 output feature (quality score between 0 and 10)

## **Project Workflow**


**Data Import & Preprocessing**

Checked missing values

Transformed quality into binary classification: good or bad

Applied Label Encoding

Feature Scaling using StandardScaler

**Exploratory Data Analysis (EDA)**

Bivariate Analysis (scatter plots, bar charts)

Correlation Heatmap

Pairplots to visualize feature relationships

**Model Building**

Logistic Regression

Stochastic Gradient Descent (SGD)

Support Vector Classifier (SVC)

Decision Tree Classifier

Random Forest Classifier

Multi-Layer Perceptron (MLP)

Artificial Neural Network (ANN with Keras)

**Model Evaluation**

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Cross-Validation

## **Technologies Used**


Language: Python 3

Libraries:

numpy, pandas

matplotlib, seaborn

scikit-learn

keras (for ANN)

Platform: Google Colab / Jupyter Notebook

## **Results**

**Best Performing Models:**

Random Forest Classifier

Artificial Neural Network (ANN)

**Key Insights:**

Alcohol content is positively correlated with wine quality.

Volatile acidity and density are negatively correlated with wine quality.

Feature selection can significantly impact model performance.

## **Key Graphs**

Correlation Heatmap

Feature importance plots

Confusion matrices

Accuracy curves during training (for ANN)
