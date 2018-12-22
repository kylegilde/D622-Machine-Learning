## DATA622 Project
- Assigned on November 13, 2018
- Due on December 11, 2018 12:00 PM EST
- 25 points possible, worth 25% of your final grade

### Instructions:
Read the online textbook [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/) in its entirety.  Use the code in the book to guide your steps to build an image recognition model on the classic MNIST dataset.  The data pull steps can also be found in the book.

### Critical Thinking (10 points)

**Submit a ~500 word explanation of the choices and tradeoffs you made in the process of building this model.  (e.g. why did you go X layers deep? why did you choose X cost function?).**

#### Introduction

From a modeling perspective, the most significant difference between deep learning/neural networks and other machine learning algorithms is the larger number of parameters. The purpose of these specifications is to enhance the learning speed, which means how the accuracy increases over a certain number of epochs, and to prevent overfitting in the training data. The purely numerical parameters are the number of hidden layers and neurons in each layer, the number of epochs, the early-stopping epoch threshold, the size of mini-batches the learning rate & regularizations. The more qualitative parameters include the types of layers/network, the types of cost and activation functions, weight initialization, dropouts and artificially creating more training samples.

Michael Nielson's web book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) provides an excellent overview on the subject and how to navigate the parameter-selection process using the MNIST dataset. For convenience, I used Anton Vladyka's home-spun [PDF version of the book](https://github.com/antonvladyka/neuralnetworksanddeeplearning.com.pdf/blob/master/book.pdf) and Michal Dobrzanski's [3.5 version of the Python code](https://github.com/MichalDanielDobrzanski/DeepLearningPython35). The following parameter-tuning exercise will perform manual grid searches on networks with quadratic/sigmoid functions and cross entropy/sigmoid functions. I will initeratively try different learning rates and regularizations on the training and validation datasets and conclude with applying our optimized parameters to the test data.

#### Adopted Parameters

Initially, during the training, I relied upon Nielson's subject matter expertise and adopted some of his default parameters employed during the first three chapters, including the number of hidden layers and neurons, mini-batch size, the early-stopping epoch threshold, weight initialization. Using only 1 hidden layer with 30 neurons yields shorter computation times during our model training. Nielson uses mini-batches of 10 samples as a "compromise value that maximizes the speed of learning" (116). This quantity affects how quickly a model learns. If it is too small, the information gained per epoch will be limited. If it is too large, then each epoch takes longer to finish, and we cannot run as many over the same period of time. In order to use one's training time efficiently, Nielson recommends beginning with an early-stopping mechanism of 10 epochs, which means that the model stops if the validation accuracy does not improve for 10 consecutive epochs (113). He also recommends increasing this period as one finds better parameters. Dobrzanski implemented this feature in his `network2.py` script. When using sigmoid neurons, initiating better random weights in the hidden neurons can help to avoid learning slowdowns. Using a more concentrated version of the normal distribution with a smaller standard deviation makes it much less likely that we start off with using slow-learning saturated weight values (96). This impacts the learning speed of the network in the short-term and improves long-run performance. Finally, I used Nielson's suggestion to use only subsets of the training and validation datasets, 5,000 and 1,000 images respectively, to enable faster parameter selection (109). This decreased the training runtimes by a factor of 10 and enabled me to test many more parameters.

#### Optimizing Quadratic-Sigmoid Network

The first network I optimized has a quadratic cost function and sigmoid activation function. I began with the learning rate. This important parameter determines the size of the steps in the stochastic gradient descent of the cost function. If it is too large, the steps will overshoot the minimum, and if it is too small, the minimum may never be reached (112). As Nielson recommends, I initialized a list of possible rates by powers of ten. I then looped through the list and used the best rate to create a smaller range of values to test. After a few iterations, I found the best learning rate for stable, long-term learning to be 0.4. I locked in this value and moved on to L2 regulatization.

Nielson explains that regularization can reduce the amount of overfitting. L2 weight decay requires a positive number, and it adds a large-weight penalty to the cost function and its derivatives. L2's primary benefit is reduced overfitting to the noise in the data. The author asserts that unregularized models seem to get "stuck" in local minima and that regularized models produce more replicable results (83). To optimize L2, I used the same method as I did with the learning rate. After a few iterations of testing smaller and more specific ranges, I found the best lambda value to be 0.005, and the accuracy on the validation dataset was 92% (see the `fig` folder for plots).

#### Optimizing Cross Entropy-Sigmoid Network

For our next network, I swapped out the quadratic function for the cross-entropy cost function and retained the sigmoid neurons. The particular cost function used in a neural network model can have a significant effect on how quickly or slowly a model learns. In chapter 3, Nielson demonstrates the advantages of using cross-entropy with the sigmoid neurons (59-69). Because the partial derivatives of the quadratic cost contain the derivative of the sigmoid function, the "learning curve" can remain flat when the initial randomized weights and biases are significantly different than the actual values. However, with cross-entropy, its partial derivatives have the advantage of not retaining the sigmoid derivative. Consequently, the cross-entropy's learning curve is much steeper than the quadratic function. The cross-entropy function requires fewer epochs to achieve the same accuracy level as the quadratic and is often able to surpass it.

After iterating through narrower and narrower ranges, I found that a learning rate of 0.056 and L2 of 0.07525 produced a 91.4% validation accuracy rate.

#### Test Data Accuracy & Conclusions

Finally, I applied our models and optimized parameters to the unseen data in the test set. For the test set runs, I used all 50,000 training images and expanded the number of epoches from 30 to 100 and the early-stopping threshold from 10 to 33, so that we could take a look at how our models perform over longer periods. In addtion to the accuracy of the predictions on the test data, we will want to check that the models did not overfit to the training data.

The quadratic-sigmoid model finished with 96.1% test accuracy, and the training accuracy exceeded the test accuracy by 2 percentage points.

The cross-entropy-sigmoid model stopped early at Epoch 84 and yielded an accuracy rate that was slightly higher than quadratic model at 96.2%. This model was more overfit than the quadratic model at nearly 3 percentage points. The accuracy of the predictions on the test set increased faster than the correct predictions on the training data set.

The plots from all of the model runs can be seen in the `fig` folder. With more time and more experience with Python classes, I would have liked to have fixed the bugs in `my_network2.py`, where I attempted to implement the ReLu activation function. Nielson explains that these neurons can potentially create better models because they do not saturate like the sigmoid neurons (124).

#### Post Script: Other Regularizations

I want to describe briefly 2 other methods that Nielson mentions but does not implement in `network2.py`. They can reduce overfitness and improve accuracy.

Dropout refers to randomly and repetitively deleting half of the hidden neurons. This creates a network model that is robust to losses in information, which can limit the amount of overfitting (89). 

Another way to train a better model is to artificially create more training data. In the case of the NMIST images, one could rotate, skew and distort the numbers in ways that reflect variations in human handwriting (92).


### Applied (20 points)
Submit all code, clean and commented.  Feel free to use any distributed computing methods I discussed in previous lessons to help you in this process, but it's not necessary.  

Please see **`my_modeling.py`** and my attempt to model with the ReLU activation function in `modeling_relu.py`

### Additional Resources

1. The [Goodfellow book](http://www.deeplearningbook.org/) on Deep Learning is the authorative resource on this subject still.  
2. Check out [this curated list](https://github.com/ChristosChristofidis/awesome-deep-learning) for even more resources on deep learning.  
