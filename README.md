## Movie Review Classification

This project looks at movie reviews and labels them either positive or negative,
using a neural network composing of an embedding layer, an averaging layer, a dense layer with a rectifying linear unit (relu for short)
activation function and an output dense layer using a sigmoid function. The data used to train the model
is taken from the keras imbd database and processed for a more accurate end result. Each time the code is run, a review
"review.txt" which is not from the database is run as well as 10 database samples displayed along with the model's
prediction and actual result. The accuracy of this model is evaluated at 88%, this is paired with the fast 1 minute run
time it take to train the model. The accuracy could be improved by further 
tweaking and longer training times.