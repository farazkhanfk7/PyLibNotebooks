import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(‘Student-Pass-Fail-Data.csv’)
df.head()

x = df.drop(‘Pass_Or_Fail’,axis = 1)
y = df.Pass_Or_Fail

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

logistic_regression = LogisticRegression()
