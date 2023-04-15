# Yelp Review Classification

This code provides an implementation of a recurrent neural network (RNN) for sentiment analysis on Yelp reviews. <br>
It first preprocesses the data, converting text reviews into numerical input vectors for the neural network. The vocabulary of the reviews is initialized, and words that occur less than a certain number of times are removed. The RNN is then trained using the input vectors and their associated labels (positive or negative sentiment).<br>
The architecture consists of an input layer, a hidden layer, and an output layer. The hidden layer uses a $tanh$ activation function, and the output layer uses a sigmoid activation function. The training is done using back-propagation a manually written gradient descent. Finally, the trained model is used to make predictions on new reviews, where the review is vectorized and passed through the trained neural network to get a prediction of its sentiment (positive or negative).

![rnn_cascade](rnn_cascade.gif)

## Preprocessing
---
Before any type of data manipulation, it's necessary to define a function that allows the neural network to accept variable length text as input.

The `vectorize()` function is used to convert a review text into a one-hot encoded matrix representation that can be inputted into the RNN model.

It first splits the input into individual tokens, filtering out any tokens that are punctuation. For each non-punctuation token, the function generates a one-hot encoded vector with a length equal to the size of the vocabulary (i.e. the number of unique tokens in the corpus) and sets the index corresponding to the token to 1.

The function then concatenates all of the generated one-hot vectors horizontally to create a matrix representation of the input review. Finally, the function returns the one-hot encoded matrix.

