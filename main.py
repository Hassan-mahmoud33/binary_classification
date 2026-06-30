import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

# to clear the terminal
def delete():
    os.system('cls' if os.name == 'nt' else 'clear')
delete()

data = pd.read_csv("ex2data1.txt" , header=None , names=['Exam 1' , 'Exam 2' , 'Admitted'] , skipinitialspace=True)

def display():
    print(f'{'-'*50}\n')
    print(data.head(3))
    print(f'{'-'*50}\n')

    print(data.describe())
    print(f'{'-'*50}\n')
    print(data.info())
    print(data.nunique())

display()

#******************************************************************************


positive = data [ data['Admitted'].isin([1])]
negative = data [ data['Admitted'] == 0 ]

# print(f"Admitted students : {positive}")
# print(f"Inadmitted students : {negative}")

# print(f"Admitted students = {(data['Admitted'] == 1 ).sum()}")
# print(f"Inadmitted students = {(data['Admitted'] == 0 ).sum()}")


#******************************************************************************

def draw_points():
    fig , ax1  = plt.subplots( figsize = (7 , 5))
    ax1.scatter(positive['Exam 1'] , positive['Exam 2'], s= 50 , marker='o', label = 'Admitted')
    ax1.scatter(negative['Exam 1'] , negative['Exam 2'], s= 50 ,c = 'r',  marker='x', label = 'Not Admitted')

    ax1.legend(loc = 1)
    ax1.set_xlabel('Exam 1')
    ax1.set_ylabel('Exam 2')
    plt.show()

draw_points()

#******************************************************************************


def sigmoid(z):
    return 1 / ( 1  + np.exp( -z ) )


# to draw sigmoid function
nums = np.arange(-10 , 10 , 1)

fig , ax = plt.subplots(figsize = ( 7 , 5))
ax.plot( nums , sigmoid(nums), 'r--')

plt.show()

#******************************************************************************


data.insert( 0 , 'ones' , 1)
# print(data.head(2))

cols = data.shape[1]
X = data.iloc[ :  , 0 : cols - 1 ]
y = data.iloc[ : , cols - 1 : cols]

# print(f"X = {X.head(4)}\n")
# print(f'{'-'*50}\n')
# print(f"y = {y.head(4)}\n")

X = np.array(X.values)
y = np.array(y.values)

# print(f"X = {X[0 : 3 ]}\n")
# print(f'{'-'*50}\n')
# print(f"y = {y[ 0 : 3 ]}\n")

theta = np.zeros(3)
# print(theta)


#***************************************************************************


def cost( theta , X , y ):

    first = np.multiply( -y , np.log( sigmoid( X.dot(theta) ) ))
    second = np.multiply( ( 1 - y) , np.log( 1 - ( sigmoid( X.dot(theta) )) ) )
    
    return np.sum( first - second ) / len(X)

cost_value = cost( theta , X , y )


#*******************************************************************************************


def gradient( theta ,  X , y ):
    
    parameters = int ( len(theta))
    grad = np.zeros(parameters)  

    error = sigmoid( X.dot(theta )) - y

    for i in range(parameters):
        term = np.multiply( error , X[ : , i ])

        # equation of gradient 1 / m * sum( h(x) - y ) x
        grad[i]= np.sum(term) / len(X)        

    return grad



#********************************************
# library that give you the lower (minimum) value for a math function
import scipy.optimize as opt

result = opt.fmin_tnc ( func=cost , x0=theta , fprime=gradient , args=( X , y) , messages=False)

# print(f"result = {result}") 
# (array([-25.16131865,   0.20623159,   0.20147149]), 36, 0)
# ( [ theta0 , theta1 , theta2 ] , iterations , 0 = optimization has successed  )


#*******************************************************************************************

cost_after_opt = cost( result[0] , X , y)

print(f"cost before optimization = {cost_value}")
print(f"cost after optimization = {cost_after_opt}")

#*******************************************************************************************


def predict( theta , X):

    probability = sigmoid( X *  theta.T )
    return [ 1 if x >= 0.50 else 0 for x in probability ]

theta_min = np.matrix(result[0])
predictions = predict(theta_min , X )

print(f"new predictions = {predictions}\n")

correct = [ 1 if (( a == 1 and b == 1 ) or ( a == 0 and b == 0 ))
           else 0  for ( a , b ) in zip(predictions , y)]

accuracy = ( sum(map( int , correct)) / len(correct) * 100 )
print(f"Accuracy = {accuracy} %")

