```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```


```python
#load train and test data
train_df = pd.read_csv(Path('2019loans.csv'))
test_df = pd.read_csv(Path('2020Q1loans.csv'))
# print(train_df.head())
# print(test_df.head())
```


```python
# Convert categorical data to numeric and separate target feature for training data

# input variables for training set
X_train = train_df.drop('loan_status', axis=1)

# target variable for training set
y_train = train_df['loan_status']
X_train = pd.get_dummies(X_train)
```


```python
# Convert categorical data to numeric and separate target feature for testing data

# input variables for testing set
X_test = test_df.drop('loan_status', axis=1)
 
    # target variable for testing set
y_test = test_df['loan_status'] 
X_test = pd.get_dummies(X_test)
```


```python
# Find the columns in the training set that are not present in the testing set
missing_cols = set(X_train.columns) - set(X_test.columns)

# add missing dummy variables to testing set
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

```


```python
# Train the Logistic Regression model on the unscaled data and print the model score
logreg = LogisticRegression(max_iter= 5000)
logreg.fit(X_train, y_train)
logreg_score = logreg.score(X_test, y_test)
print(f"Logistic Regression Model Score: {logreg_score:.4f}")
```

    Logistic Regression Model Score: 0.5587
    

    C:\Users\jcrys\anaconda3new\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    


```python
# Train a Random Forest Classifier model and print the model score

# fit random forest classifier model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_score = rfc.score(X_test, y_test)
print(f"Random Forest Classifier Model Score: {rfc_score:.4f}")
```

    Random Forest Classifier Model Score: 0.6306
    


```python
# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
# Train the Logistic Regression model on the scaled data and print the model score
lr = LogisticRegression(max_iter= 5000)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_score_scaled = lr.score(X_test_scaled, y_test)
print(f"Logistic Regression Model Score: {lr_score_scaled:.4f}")
```

    Logistic Regression Model Score: 0.6913
    


```python
# # Train a Random Forest Classifier model on the scaled data and print the model score
rfc.fit(X_train_scaled, y_train)
rfc_score_scaled = rfc.score(X_test_scaled, y_test)
print(f"Random Forest Classifier Model Score: {rfc_score_scaled:.4f}")
```

    Random Forest Classifier Model Score: 0.7964
    


```python

```
