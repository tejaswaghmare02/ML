#program for implementation of decision tree

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#import dataset
data=pd.read_csv('glass.csv')
print(data.head(5))

#split dataset into features and target
x=data.iloc[:,2:-1].values
y=data.iloc[:,-1].values

#split training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3,random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#train the model
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

#predict the model
y_pred=model.predict(x_test)

#check accuracy
print("Accuracy:",accuracy_score(y_test,y_pred)*100)

#check classification report
print(classification_report(y_test,y_pred))

#print confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix :\n",cm)

#plot confusion matrix
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt="g")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#visualize the tree
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model,filled=True)
plt.show()
