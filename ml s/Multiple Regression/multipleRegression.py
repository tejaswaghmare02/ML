import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


cardata=pd.read_csv('car.csv')
print(cardata.head(5))

X=cardata.iloc[:,[2,3]].values
y=cardata.iloc[:,-1].values

x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=1/4,random_state=10)

reg=LinearRegression()
model=reg.fit(x_train,y_train)
prediction=model.predict(x_test)

#print training and testing scores
print("Training Score is ",model.score(x_train,y_train))
print("Testing Score is ",model.score(x_test,y_test))

#Plot the graph
ax = plt.axes(projection ="3d")
ax.scatter3D(x_train[:,0],x_train[:,1],y_train,color="red",label="Training")
ax.scatter3D(x_test[:,0],x_test[:,1],y_test,color="blue",label="Testing")
ax.set_xlabel("Volume")
ax.set_ylabel("Weight")
ax.set_zlabel("CO2")
ax.legend(loc="lower right")
plt.show()
