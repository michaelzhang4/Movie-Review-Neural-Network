from tensorflow import keras
import numpy as np
import random

data = keras.datasets.imdb  # import imdb reviews data with positive negative review labels

(train_data, train_labels), (test_data, test_labels) = data.load_data(
    num_words=88000)  # getting the data with 88000 most common words into training/test data and labels

word_index = data.get_word_index()  # gets an index mapping for each word

word_index = {k: (v + 3) for k, v in word_index.items()}  # moves the index of each word up 3 spaces
# each index mapping is moved up so these mapping can be added
word_index["<PAD>"] = 0  # PAD used to make up empty space in a review
word_index["<START>"] = 1  # START used to indicate beginning of a review
word_index["<UNK>"] = 2  # UNK used to indicate unknown word in a review
word_index["<UNUSED>"] = 3  # UNUSED used to indicate unused word in a review

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # creates a reversed dictionary of key/value to be used for decoding reviews

# decodes integer-encoded reviews into english
def decode_review(text):
    review = []
    for i in text:
        word = reverse_word_index.get(i,"?")
        if word != "<PAD>":
            review.append(word)
    return " ".join(review)
    # return " ".join([reverse_word_index.get(i, "?") for i in text])

# encodes english-encoded reviews into integers
def review_encode(review):
    encoded = [1]
    for word in review:
        encoded.append(word_index.get(word.lower(), "2"))
    return encoded

# takes the average of all word prediction values and returns a positive review if the average is 50% or above and negative otherwise
def aggregate_predict(predict):
    if np.mean(predict) >= 0.5:
        return 1
    else:
        return 0


# tried but didn't work, use keras for preprocessing instead
# def normalise_data_length(data_input, length):
#     array = np.empty(len(data_input), )
#     for x in data_input:
#         if len(x) >= length:  # pops off all values in list over the set length
#             for y in range(0, (len(x) - length)):
#                 x.pop()
#         else:  # appends 0s to fill up list to set length
#             for y in range(0, length - len(x)):
#                 x.append(0)
#         np.append(array, np.array(x))  # changes type of list to a numpy array and appends it to the overarching array
#     return array  # returns numpy array that holds length of input array, numpy arrays
#

# preprocessing data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)  # normalise length of train data to 250

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                       maxlen=250)  # normalise length of test data to 250

# model

# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000, 16))  # establishes dense vectors of 16 dimensions for each word
# model.add(
#     keras.layers.GlobalAveragePooling1D())  # finds the average of each dimension of all vectors and returns as a list of 16 averages
# model.add(keras.layers.Dense(16,
#                              activation="relu"))  # makes negative components in each vector 0 and computes positive components
# model.add(keras.layers.Dense(1,
#                              activation="sigmoid"))  # returns a value between 0 and 1 indicating positive or negative review
#
# model.summary()     # prints summary of model, layers and parameters
#
# model.compile(optimizer="adam", loss="binary_crossentropy",
#               metrics=["accuracy"])   # configure training related settings and display accuracy as output
#
# val_data = train_data[:10000]   # splitting training data into validation and training sets
# training_data = train_data[10000:]
#
# val_labels = train_labels[:10000]   # splitting training labels into validation and training sets
# training_labels = train_labels[10000:]
#
# model.fit(training_data, training_labels, epochs=40, batch_size=512,
#           validation_data=(val_data, val_labels), verbose=1)    # fitting the model using 40 runs,
# # each with a data input size of 512 taken from training data and checked against validation data to prevent overfitting
#
# results = model.evaluate(test_data, test_labels)
#
# print(results)

# model.save("model.h5")    # saves model so model doesn't need to be trained every time the program is run

model = keras.models.load_model("model.h5")  # loads in pre-trained model

# processes review from review.txt and prints out the model's prediction of whether it is a positive or negative review
with open("review.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(";", "").replace(
            ")", "").replace(":", "").replace(".", "").replace(
            "(", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
        predict = model.predict(encode)
        print("\nReview taken from internet: ")
        print("<START> " + line)
        print("Prediction: " + str(aggregate_predict(predict)) + "\nActual: 1")

# prints out 10 randoms reviews, with the prediction and the actual result
for counter in range(0, 10):
    ran = random.randint(0, 25000)
    test_review = test_data[ran]
    predict = model.predict([test_review])
    print("\nReview " + str(counter + 1) + ": \n" + decode_review(test_review))
    print("Prediction: " + str(aggregate_predict(predict)) + "\nActual: " + str(test_labels[ran]))
