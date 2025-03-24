import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("petrol_consumption.csv")
# print(dataset.info())

X = dataset[["Petrol_tax","Average_income","Paved_Highways","Population_Driver_licence(%)",]]
Y = dataset["Petrol_Consumption"]

X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(Y_test)

Y_pred = regressor.predict(X_test)
df = pd.DataFrame({"Actual:" :Y_test," Predicted:":Y_pred})

print(df)
