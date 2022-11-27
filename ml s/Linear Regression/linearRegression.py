from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

salary=pd.read_csv('salary.csv')
x=salary['YearsExperience'].tolist()
y=salary['Salary'].tolist()


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=10)

slope, intercept, r, p, std_err =stats.linregress(x,y)

print("Slope={}\nIntercept={}\nCoefficient of correlation={}\npValue={}\nStandard Error={}\n".format(slope,intercept,r,p,std_err))


#function to calculate y for different x values based on given data
def sol_fun(x):
    return slope*x+intercept 

#model to plot regression line 
reg_model=list(map(sol_fun,x_train))

#scatter plot of dataset

plt.scatter(x_train,y_train, color="red",label="Training")
plt.scatter(x_test,y_test,color='blue',label="Testing")


#regression line
plt.plot(x_train,reg_model,color="Green")
plt.title("Salary vs Experience")
plt.ylabel("Salary")
plt.xlabel("Years of Experience")
plt.legend(loc="lower right")
plt.show()


#prediction
x_var=float(input("Enter years of experience: "))
print("Predicted Salary Amount is",round(sol_fun(x_var),2))

