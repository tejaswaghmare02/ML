#program to implement random forest algorithm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#read the data
data = pd.read_csv('wine.csv')
print(data.head(5))

#assign features and targets
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#split the data into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#train the model
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)

#predict the model
y_pred=model.predict(x_test)

#evaluate the model
cm=confusion_matrix(y_test,y_pred)
print('\nAccuracy Score:',accuracy_score(y_test,y_pred))
print('\nClassification Report:',classification_report(y_test,y_pred))
print('\nConfusion Matrix:\n',cm)

#plot confusion matrix
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#visualizing the tree
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model.estimators_[0],filled=True)
plt.show()



