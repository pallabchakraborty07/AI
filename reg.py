import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


ages = np.array([5,6,7,8,9,10,11,12])
heights = np.array([3,4,5,2,4,3,5,2])

model = LinearRegression()
model.fit(ages,heights)

predicted_height = model.predict(ages)

plt.scatter(ages,heights,color="blue",label="Actual Heights")
plt.plot(ages,predicted_height,color="red",label="Regression Line")

plt.xlabel("age")
plt.ylabel("heights")
plt.show()