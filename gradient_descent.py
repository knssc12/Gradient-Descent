import numpy as np
import matplotlib.pyplot as plt
import csv

results = []
a = []
b= []

with open('2017.csv') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
myarray = np.asarray(results)

for i in range(1, 140):
	a.append(myarray[i,3])
	b.append(myarray[i,6])

X_U32 = np.asarray(a)
Y_U32 = np.asarray(b)
X = X_U32.astype('float64')
Y = Y_U32.astype('float64')

#the length of the dataset
n = float(len(X))
#initial m,c
m = 0
c = 0
#learning rate
L = 0.0001

#epochs
epochs = 1000

for i in range(epochs):
	Y_pred = m*X + c
	D_m = (-2/n) * sum((Y - Y_pred)*X)
	D_c = (-2/n) * sum(Y - Y_pred)
	m = m - L*D_m
	c = c - L*D_c
	w = Y - Y_pred
	error = (1/n) * sum((Y-Y_pred)**2)
	print(error)
# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()