#program to implement svm for non linear data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

data=pd.read_csv("user.csv")
print(data.head(5))

#splitting the data into features and target

x=data.iloc[:,[2,3]].values
y=data.iloc[:,-1].values

#splitting the data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#creating the model
model=svm.SVC(kernel='rbf',random_state=0)
model.fit(x_train,y_train)

#predicting the values
y_pred=model.predict(x_test)

#classification report
print("\nClassification Report:\n",classification_report(y_test,y_pred))

#calculating the accuracy
print("\nAccuracy:",accuracy_score(y_test,y_pred))

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",cm)

#plot confusion matrix
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#print roc_auc_score
print("\nROC_AUC_SCORE:",roc_auc_score(y_test,y_pred))

#plotting the roc curve
from sklearn.metrics import roc_curve, auc

# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class, not the predicted outputs.
y_train_pred = model.decision_function(x_train)    
y_test_pred = model.decision_function(x_test)

# Compute ROC curve and ROC area for each class
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

#plot
plt.grid()
plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()

#visualizing the training set results
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                    np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                alpha=0.75,cmap=ListedColormap(('red','blue')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','yellow'))(i),label=j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()