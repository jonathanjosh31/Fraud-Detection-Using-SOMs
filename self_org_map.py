
'''

Building a Self organizing maps

for detecting Frauds in a particular dataset.

'''

#Importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the required dataset
org_dataset = pd.read_csv('Credit_Card_Applications.csv')

#Spliting into subsets
x = org_dataset.iloc[:,:-1].values
y= org_dataset.iloc[:,-1].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)  #fitting the dataset to the minmaxscaler object to perform normalisation.

#Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len = 15, sigma=1.0 , learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100) 