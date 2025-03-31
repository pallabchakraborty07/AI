import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

Y= np.array ([0,1,0,1,1,1,0,0,0,1])
X= np.arange(10).reshape(-1,1)

model = LogisticRegression(C=10.0,solver="liblinear")
model.fit(X,Y)

prob_predict = model.predict_proba (X)
Y_pred = model.predict(X)

score=model.score(X,Y)
report = classification_report(Y,Y_pred)

print(score)
print(report)

