#Program to implement KNN classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

#Load the dataset
data = pd.read_csv('diabetes.csv')
print(data.head(10))
#separate data
x=data.iloc[:, [4,5]].astype(int).values
y=data.iloc[:,8].astype(int).values

#Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors = 7)

#Fit the classifier to the data
knn.fit(x_train, y_train)

#Predict the response for test dataset
pred = knn.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("\nAccuracy:",accuracy_score(y_test, pred))

#creating confusion matrix
cm = confusion_matrix(y_test, pred)
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
#classification report
print(classification_report(y_test, pred))

#plotting decision region
plot_decision_regions(x_train, y_train, clf=knn, legend=2)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("KNN classification")
plt.show()