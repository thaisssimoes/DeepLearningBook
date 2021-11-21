import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt

"""
X is used to represent input into the model
y is label of the expected output
"""
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

#Shows the output
print(X_train.shape)

#Shows how many labels we have
print(y_train.shape)

"""
Plot a img of the first 12 digits from the dataset in a 5x5 scale on both axis, in grey color
"""
# plt.figure(figsize=(5, 5))
# for k in range(12):
#     plt.subplot(3, 4, k+1)
#     plt.imshow(X_train[k], cmap="Greys")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

"""
Flattening two-dimensional data into one dimension. For that, we use the X values in our code

numpy.reshape
"""
X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')

"""
Guaranteeing the values ranging between 0 and 1 
"""
X_train = X_train/255
X_valid = X_valid/255



"""
Converting integer labels to one-hot

n_classes = total number of possible outcomes
y_train / y_valid are used to validate data.

Example of how the number 7 validation array would look like after the application of the to_categorical function:

array([0.,0.,0.,0.,0.,0.,0.,1.,0.,0.])
"""
n_classes = 10
y_train = keras.utils.np_utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.np_utils.to_categorical(y_valid, n_classes)


"""Keras code to architect a shallow NN"""
model = Sequential()

#Creates the hidden layer with 64 nodes, using the sigmoid function and receiving input from 784 entries
model.add(Dense(64, activation='sigmoid', input_shape=(784, )))
#creates the decision layer with 10 possible outputs, using softmax activation.
#Softmax is used to provide non-binary classification
model.add(Dense(10, activation="softmax"))

model.compile('SGD', loss='mean_absolute_error', metrics=['mean_absolute_error'])

model.fit(x=X_train, y=y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_valid, y_valid))










