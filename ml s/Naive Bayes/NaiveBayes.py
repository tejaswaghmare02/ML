#program for implementation of naive bayes classifier

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

#loading the dataset
data=pd.read_csv('user.csv')
print(data.head())

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

#splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3,random_state=0)

#fitting the naive bayes classifier to the training set
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#predicting the test set results
y_pred=classifier.predict(x_test)

#Accuracy
print("Accuracy:",accuracy_score(y_test, y_pred))

#Accuracy report 
print("\n",classification_report(y_test, y_pred))

#making the confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion matrix\n\n",cm)

#visualizing confusion matrix
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
ax.set_title('\nConfusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

#plotting decision region
plot_decision_regions(x_train, y_train, clf=classifier, legend=2)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Naive Bayes Classification")
plt.show()

