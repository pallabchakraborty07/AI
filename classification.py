import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# dataset = load_breast_cancer()

# sb.set_style('dark')
# plt.figure(figsize=(8,10))
# (unique,counts) = np.unique(dataset['target'],return_counts=True)
# sb.barplot(x=dataset['target_name'],y=counts)
# plt.show()

data ={
"age":[10,11,12,13,14,15],
"height":[4,4.2,4.5,5,5.1,5.3],
"class":[5,6,7,8,9,10],
}

dataset = pd.DataFrame(data)

X = dataset[["age","height"]]
Y = dataset["class"]

#Data Collect-Data Category-Training & Test
x_train ,x_test, y_train , y_test , = train_test_split(X,Y,test_size=0.2)

#Model
Scaler = StandardScaler()

x_train = Scaler.fit_transform(x_train)
x_test = Scaler.transform(x_test)
clsf = RandomForestClassifier()

clsf.fit(x_train,y_train)
Y_pred = clsf.predict(x_test)

print(Y_pred)