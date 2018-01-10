
# Give Me Some Credit

This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

The goal of this competition is to build a model that borrowers can use to help make the best financial decisions.

There are a total of 4 notebooks for this project in the order of:
1. [Data Cleaning](https://github.com/joannasys/Kaggle---Credit/blob/master/1_Data%20Cleaning.ipynb)
2. [EDA (Exploratory Data Analysis)](https://github.com/joannasys/Kaggle---Credit/blob/master/2_EDA.ipynb)
3. [Data Preprocessing](https://github.com/joannasys/Kaggle---Credit/blob/master/3_Data%20Preprocessing.ipynb)
4. [Data Modeling](https://github.com/joannasys/Kaggle---Credit/blob/master/4_Data%20Modeling.ipynb)

Note: You may run the notebooks in the order above to reproduce the same results. .csv files in the Submissions folder will already be in the format for Kaggle submission. 

Here's a quick summary on the project workflow:

<img src='https://i.imgur.com/PZYKh0L.png'>

## Understanding the data

The very first step to a machine learning workflow is to understand:
1. The problem that you are trying to solve
2. The objective of the project
3. The data you have

The problem we are trying to solve here is that borrowers do not have a good understanding of whether they are in the right position to continue borrowing. The objective of the project is to predict the probability of an individual experiencing financial distress in the next 2 years based on his or her current financial state.

Traditionally, banks uses credit scoring to analyse an individual's creditworthiness. A quick desk research provided the following information:

Biggest 5 factors that affects credit scoring:
1. Payment History – 35% of credit score
2. Amounts Owed – 30% of credit score
3. Length of Credit History – 15% of credit score
4. New Credit – 10% of credit score
5. Types of Credit In Use – 10% of credit score

Source: https://www.investopedia.com/articles/pf/10/credit-score-factors.asp

Code in [Data Cleaning](https://github.com/joannasys/Kaggle---Credit/blob/master/1_Data%20Cleaning.ipynb)

#### Data Dictionary

<div class="output_subarea output_html rendered_html output_result"><div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable Name</th>
      <th>Description</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SeriousDlqin2yrs</td>
      <td>Person experienced 90 days past due delinquency or worse</td>
      <td>Y/N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RevolvingUtilizationOfUnsecuredLines</td>
      <td>Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits</td>
      <td>percentage</td>
    </tr>
    <tr>
      <th>2</th>
      <td>age</td>
      <td>Age of borrower in years</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NumberOfTime30-59DaysPastDueNotWorse</td>
      <td>Number of times borrower has been 30-59 days past due but no worse in the last 2 years.</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DebtRatio</td>
      <td>Monthly debt payments, alimony,living costs divided by monthy gross income</td>
      <td>percentage</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MonthlyIncome</td>
      <td>Monthly income</td>
      <td>real</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NumberOfOpenCreditLinesAndLoans</td>
      <td>Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumberOfTimes90DaysLate</td>
      <td>Number of times borrower has been 90 days or more past due.</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NumberRealEstateLoansOrLines</td>
      <td>Number of mortgage and real estate loans including home equity lines of credit</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NumberOfTime60-89DaysPastDueNotWorse</td>
      <td>Number of times borrower has been 60-89 days past due but no worse in the last 2 years.</td>
      <td>integer</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NumberOfDependents</td>
      <td>Number of dependents in family excluding themselves (spouse, children etc.)</td>
      <td>integer</td>
    </tr>
  </tbody>
</table>
</div></div>

A quick look at our data dictionary, we have columns that are under the category of Payment History i.e. NumberOfTime30-59DaysPastDueNotWorse, and Amounts Owed i.e. RevolvingUtilizationOfUnsecuredLines - which is the top 2 factors that affects credit scoring.

## Cleaning the data

There are a total of 150000 rows in our dataset. However, there are some NaN values in some columns. NaN values will be problematic as most algorithms do not work well with empty cells and would result in error. 

There are many ways to deal with NaN values: 
1. Drop the rows/columns entirely
2. Impute NaN values with
    * Mean or Median of the column
    * Other imputation methods such as KNN or SVD

Code in [Data Cleaning](https://github.com/joannasys/Kaggle---Credit/blob/master/1_Data%20Cleaning.ipynb)

## Analyse the data

Deep dive into each column of our data. What do our data tell us? 

In summary, we found out that there are some extreme outliers in our dataset that would seem to be abnormal in the real world. Such abnormal data would cause skewness in our data distribution. Instead of dropping the rows completely (except for some really extreme ones), I replaced them with the median of the column so that we can still preseve some of the information from other columns.

We also found out that Utilization Of Unsecured Lines has the most predictive power in this dataset. This information correspond with the top 2 factors that affect credit scoring.

Have a look in details at notebook: [EDA (Exploratory Data Analysis)](https://github.com/joannasys/Kaggle---Credit/blob/master/2_EDA.ipynb)

## Prepare the data

### Train Test Split

Train Test Split is essential to prevent overfitting of our data. I will set aside 30% of our training data as our test set. Even though we have a large number of dataset, imbalanced class is present. Hence, we would use the cross validation method on the other 70% of the dataset for training.

<img src = 'http://www.ds100.org/sp17/assets/notebooks/linear_regression/train_test_split.png'><br>
Source: http://www.ds100.org/sp17/assets/notebooks/linear_regression/train_test_split.png

### Data Scaling

Based on our EDA, there are outliers present in our data set. Hence, I have chosen to use Robust Scaler to scale the data. Robust scaler centers and scale the data based on percentiles thus it would not be largely affected by extreme outliers, yet at the same time still preserves data distribution.

### Imbalanced Dataset Resampling

We clearly have an imbalanced dataset as the number of positive class is only 6.7% (Positive (1) class being the person experienced 90 days past due delinquency)

<img src="https://i.imgur.com/wXDSppu.png?1">

There are various ways to resample an imbalanced dataset:
1. SMOTEEN
2. Undersampling majority class
3. Oversampling minority class
4. Cost Sensitivity
5. Easy ensemble / Balance Cascade

imblearn is a library that consist of various ways on resampling methods. This paper https://arxiv.org/pdf/1608.06048.pdf provides a good visualisation and overview of various resampling methods and how they affect scoring of a model. 

Code in [Data Preprocessing](https://github.com/joannasys/Kaggle---Credit/blob/master/3_Data%20Preprocessing.ipynb)

# Modeling

### Evaluation Metric

Why is AUC chosen to be the evalutation metric?

In a binary classification, there will be 4 outcomes from our results:
1. True Positive - Predicted positive when it is positive in fact
2. True Negative - Predicted negative when it is negative in fact
3. False Positive - Predicted positive when it is negative in fact (false alarm)
4. False Negative - Predicted negative when it is positive in fact (failed to detect financial distress)

We have a very imbalanced dataset with only 6.7% positive result. If we create a dumb classifier that only predicts negative class, we will achieve accuracy of 93.3% without doing much! Hence, accuracy will not be a good evaluation metric for this project. We want a metric that evaluates the performance of the positive and the negative classes.

AUC on the other hand evaluates the classifier as compared to random choice (AUC = 0.5). TPR = ratio of true positive to all predicted positives, FPR = ratio of false positives to all predicted negatives. It measures the trade off between TPR and FPR along a range of threshold.

### Feature Selection

I have used decision tree to find out the feature importance of each columns. With that, I used it as weights by multipying it into the values. 

## Models used

### Logistic Regression

Training Score: <b>0.836</b>
<br>Test Score: <b>0.933</b>
<br>Kaggle Submission Score: (below)

<img src="https://i.imgur.com/weWdohG.png">

### Random Forest Classifier

Training Score: <b>0.928</b>
<br>Test Score: <b>0.838</b>
<br>Kaggle Submission Score: (below)

<img src="https://i.imgur.com/fdYGKdb.png">

### XGBoost

Training Score: <b>0.872</b>
<br>Test Score: <b>0.868</b>
<br>Kaggle Submission Score: (below)

<img src="https://i.imgur.com/UV98JuJ.png">

Code in [Data Modeling](https://github.com/joannasys/Kaggle---Credit/blob/master/4_Data%20Modeling.ipynb)

## If I had more resources, I would try...

1. Imputing NaN values / outliers with imputation methods such as KNN. There is a useful library - FancyImpute that has many different imputation methods. I would love to read more into the research papers behind it. 
2. Spend more time analysing and understanding the data.
3. Try out various resampling methods and visualise how it affects the classification of our target variable.
4. Use SKLearn's RFECV (Recursive feature elimination with cross-validation) for feature selection.
5. Try out anomaly detection methods (such as isolation forest, one class SVM).


```python

```
