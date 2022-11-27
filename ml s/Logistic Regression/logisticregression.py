
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
data=pd.read_csv("autos1.csv")

#replace fuel type with 0 and 1
data=data.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':1,'LPG':0}})
print(data.head(5))

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

#splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=10)

#feature scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#fitting the model
model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)

#Actual values
print("\nActual values are :\n",y_test)

#print the prediction
print("\nPrediction is \n ",prediction)

#print accuracy score
from sklearn.metrics import accuracy_score
print("\nAccuracy score is ",round(accuracy_score(y_test,prediction),4))

cm=confusion_matrix(y_test,prediction)
print("\nConfusion matrix is given as :\n", cm, "\n")

ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
ax.set_title('\nConfusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()