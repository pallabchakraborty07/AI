import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("age.csv")
# print(data.info())

plt.scatter(data["age"],data["height"],marker="o",color="red")
# plt.show()

X = data["age"].values.reshape(-1,1)
Y= data["height"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

model = LinearRegression()
model.fit(X_train, Y_train)

Pred_Y = model.predict(X_test)
score = model.score(X_test, Y_test)

print(X_test)
print (Pred_Y)
print(score)
