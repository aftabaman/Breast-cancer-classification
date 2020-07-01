import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing the dataset
data = pd.read_csv("breast_cancer.csv")

#visualizing data set

p = sns.countplot(x='Clump Thickness', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

p = sns.countplot(x='Uniformity of Cell Size', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

p = sns.countplot(x='Uniformity of Cell Shape', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

p = sns.countplot(x='Marginal Adhesion', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

p = sns.countplot(x='Single Epithelial Cell Size', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

p = sns.countplot(x='Bare Nuclei', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

p = sns.countplot(x='Bland Chromatin', data = data, hue='Class', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
#split the data in to test and train set

from sklearn.model_selection import train_test_split

x_train , x_test ,y_train ,y_test=train_test_split(x,y,test_size=.25,random_state=0)

#feature scaling the data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#training and prediction
from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression(random_state=0)
model=regressor.fit(x_train,y_train)
y_predict=model.predict(x_test)

#getting the confusion matrix

from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_predict)

#evaluating accuracy of the model
from sklearn.model_selection import cross_val_score

accuracy= cross_val_score(estimator= regressor , X=x_train,y=y_train,cv=10)
print("Accuracy: {:.2f} %".format(accuracy.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracy.std()*100))

