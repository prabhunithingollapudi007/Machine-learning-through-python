#This model predicts hand written digits applying svm algorithm using sklearn module



#importing data bases

from sklearn.datasets import load_digits

#assigning data to digit variable

digit=load_digits()

#printing the type of data

print(digit.DESCR)
print(digit.target_names)

#checking for match of data types of input and output

print(digit.data.shape)
print(digit.target.shape)
print(type(digit.data))

#assigning inputs and outputs

x=digit.data
y=digit.target

#importing svm algorithm

from sklearn import svm

clf=svm.SVC(gamma=.001, C=100)
clf.fit(x,y)
print('Prediction')
print(clf.predict([x[2],x[-1]]))
print('Actual ',y[2],y[-1])
