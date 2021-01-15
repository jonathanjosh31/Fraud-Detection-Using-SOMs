
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

#Visualisation of the results
from pylab import bone,pcolor,colorbar,plot,show
bone()  # for opening a seperate window
pcolor(som.distance_map().T)  #getting all the MID of the wining nodes into the pcolor functions so as to  differentiate with different colors.
colorbar() #to get the legends
marker = ['o','s']
color = ['r','g']
for i, x in enumerate(x):
        w = som.winner(x)
        plot(w[0]+0.5,
             w[1]+0.5,
             marker[y[i]],
             markeredgecolor=color[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2) 
show()


#Finding Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)




