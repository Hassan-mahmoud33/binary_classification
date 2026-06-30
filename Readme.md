# Binary Classification Project
This project implements binary classification , it contains two features 
- Exam 1 , Exam 2 ( X1 , X2 independent variables )
- one target ( Admitted ) represents in ( 0 , 1 ) ( y dependent variable )

## Features
- shows information about the data
- splits the dataset to Admitted and Not Admitted
- draws the sigmoid function
- calculates mathematically sigmoid function 
- splits X ( features ) and y ( target )
- calculates mathematically Cost function
- implements manually Gradient Descent 
- optimizes the model parameters using Scipy optimization to minimize the cost function
- setups the target zero or one
- predicting new predictions
- calculates Model's Accuracy

## Data Structure
Exam 1 : the score in the first exam
Exam 2 : the score in the second exam
Admitted : Admission result ( 1 = Admitted , 0 = Not Admitted )

# Implementation details
- clear the terminal
- load the data and displays its information
- splits Admitted Student and Not Admitted Student
- shows a figure to Admitted Student and Not Admitted Student
- implements sigmoid function
  mathematical equation of sigmoid = 1 / 1 + e^ - X * theta
- draw sigmoid function
- inserts the column of the ones
- splits X features and y target
- establish theta
- calculates mathematically cost function
- implements manualy Gradient Descent
- optimizes the model parameters using Scipy optimization to minimize the cost function
- setups the target zero or one
- predicting new predictions
- calculates Model's Accuracy

  ## Result
  


- <class 'pandas.DataFrame'>
- RangeIndex: 100 entries, 0 to 99
- Data columns (total 3 columns):
- dtypes: float64(2), int64(1)
- memory usage: 2.5 KB
- Exam 1      100
- Exam 2      100
- Admitted      2
- dtype: int64
  
- Admitted students :
  
- Exam 1     Exam 2  Admitted
- 3  60.182599  86.308552         1
- 4  79.032736  75.344376         1
  
- Inadmitted students :
  
- Exam 1     Exam 2  Admitted
- 0  34.623660  78.024693         0
- 1  30.286711  43.894998         0
- Admitted students = 60
- Inadmitted students = 40
  
- result = (array([-25.16131863,   0.20623159,   0.20147149]), 36, 0)
- 
- cost before optimization = 0.6931471805599453
- cost after optimization = 0.20349770158947458
- new predictions = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, - 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, - 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]

- Accuracy = 89.0 %


## Visualization
- the figure below shows Admitted and Not Admitted Students
!(Admitted and Not Admitted Students)[images/students.png]

- the figure below shows sigmoid function distribution
!(Sigmoid)[images/sigmoid.png]

















