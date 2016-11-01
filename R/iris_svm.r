#The first time through, you'll need to install the LIBSVM package:
#install.packages('e1071', dependencies = TRUE);

# Include the LIBSVM package
library (e1071)

# Load our old friend, the Iris data set
# Note that it is included in the default datasets library
library(datasets)
data(iris)

# For your assignment, you'll need to read from a CSV file.
# Conveniently, there is a read.csv() function that can be used like so:
# myDataSet = read.csv(fileName, head=TRUE, sep=",")


# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows = 1:nrow(iris)
testRows = sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
iris_test = iris[testRows,]

# The training set contains all the other rows
iris_train = iris[-testRows,]

# Train an SVM model
# Tell it the attribute to predict vs the attributes to use in the prediction,
#  the training data to use, and the kernal to use, along with its hyperparameters.
#  Please note that "Species~." contains a tilde character, rather than a minus
model = svm(Species~., data = iris_train, kernel="radial", gamma = 0.001, cost = 10, cross = 10)

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction = predict(model, iris_test[,-5])

# Produce a confusion matrix
confusion_matrix = table(pred = prediction, true = iris_test$Species)

# Calculate the accuracy, by checking the cases that the targets agreed
agreement = prediction == iris_test$Species
accuracy = prop.table(table(agreement))

# Print our results to the screen
print(confusion_matrix)
print(accuracy)
