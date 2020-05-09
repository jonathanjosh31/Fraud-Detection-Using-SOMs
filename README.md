# Fraud-Detection-Using-SOMs
This repository is based on Self-Organizing Maps in which it detects potentials frauds on a dataset of customer applications in a bank for applying for a credit card.



Building a SOM :

Self organizing Map - An unsupervised deep learning model.

Implementation:

Here I have implemented a Self-Organizing Map in Python in which it solves a business problem(i.e Fraud Detection) using a particular dataset.

Say we have the details of the customers who are applying for an advanced credit card in a bank.
Here the SOM tries to detet potential fraud based on the information given.

Since we are using an unsupervised deep learning model we will identify some patterns in a high-dimensional dataset with non-linear relationships.
And here one of these patterns will be potential frauds.

Stage - 1 :
Importing our essential libraries such as
1)Numpy
2)Matplotlib.pyplot
3)pandas

Stage -2 :
Importing the required dataset

The data used in this repository is actually taken from the UCI Machine Learning Repo
This dataset is actually known as the Statlog(Australian Credit Approval) Dataset.

For more info on this dataset visit:

Short Summary on how this actually works:
	This dataset is about credit card applications.It's attributes have been changed to meaningless symbols for privacy and protection.
Here we have totally 14 attributes in which 6 are numerical and 8 are categorical.

Note : Here our deep learning model makes a customer segmentation in our dataset so that one of the segments will contain customers who potentially cheated.
All these customer info are the inputs for our network.Here these input points are going to be mapped in the output space and in-between them we will have
a neural network composed of neurons.Initialisation of vector of weights of the neurons are of the same size of the vector of the customer.

Hence for each customer,outpu for that customer will be the neuron which is closest to that customer and this neuron is called "The winning node" for each customer.
For each customer the winning node will be the neuron which is basically similar to that customer.Then we use neighbourhood functions to update the weights to the 
neighbours of the winning node to move them closer to the point.Each time we repeat this to every customer again and again,the output space decreases and loses dimensions until
it reaches to the point where the output space stops decreasing.So that will be the moment when we obtain our SOM in 2D with all the winning nodes are identified.

So now we will be getting closer to the frauds.Since a fraud is not based on general rules,they will actually be the outlying neurons in this 2D SOM as these outlying neurons
are far from the major neurons that actually follow the rules.

To detect these outlying neurons we would need the MEAN INTERNEURON DISTANCE.For each neuron we are going to find the the mean of the euclidean distance between that neuron 
and the neurons in its neighbourhood that we will define.Then after finding the outlying neurons we will use inverse mapping functions to identify the the customers who are
associatd with these winning nodes.

Stage - 3:
Creating Subsets.

Now we split the datasets into 2 subsets where one(x) will have all the attributes of the customer and the other(y) will have the info of whether the application of the customer 
has been approved or not.

Note:
Since we are using unsupervised deep learning here and not having values that will either 0 or 1 for each customer we will only use 'x' dataset during training .i.e no dependent 
variable is considered here.

Stage - 4:
Feature Scaling

We do it because since this is a high dimensional dataset with a lot of non-linear relationships we will require high computations here and to make it eaier we use feaature
scaling here.

Stage - 5 :
Training the Som

For this we will be using the minisom.A licensed open source SOM implementation.

Import minisom first.
Then declaring the som itself by creating a class object from the library we imported which is MiniSom.
For a more precise on ,we can use a bigger map but for a visualisation process we can use a moderate one.

MiniSom(x(rows) ,y(cols) ,input_len(no of features in x) ,sigma(radius of diff neighbourhoods in the grid) , learning_rate(decides how much the weights are updated during each iteration),decay_function(can be used to improve the convergence)).

Higher the learning rate,faster will be the convergence and lower the learning rate the longer the SOM will take time to be built.
Learning rate default value is 0.5.
In this model we are not gonna use decay_fuction,hence it's gonna be None.

After creating a class object,now we have to randomly declare the weights for our SOM.

After that we will be using a method called the "Train Random" on the SOM for it to train.
 
A method of class MiniSom
train_random(data(which is x,the dataset needed to be trained),num_iteration(number of iterations to be repeated))

we are gonna use 100 iterations.

And now after executing this without any errors,our SOM is now trained.

By completing this training, our required patterns are now identified on the SOM ( :-) ).

Stage - 6 :
Visualisation of the patterns by plotting the SOM








Projecct Status : Ongoing





