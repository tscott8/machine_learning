library (e1071)
letters = read.csv('../datasets/letters.csv', header = TRUE)
allRows = 1:nrow(letters)
testRows = sample(allRows, trunc(length(allRows) * 0.3))
letters_test = letters[testRows,]
letters_train = letters[-testRows,]
model = svm(letter~., data = letters_train, kernel="radial", gamma = .1, cost = 100, cross = 10)
prediction = predict(model, letters_test[,-1])
confusion_matrix = table(pred = prediction, true = letters_test$letter)
agreement = prediction == letters_test$letter
accuracy = prop.table(table(agreement))

#print(confusion_matrix)
print(accuracy)
#plot(model, letters)
tuned <- tune.svm(letter~., data = letters_train, gamma = 10^(-6:-1), cost = 10^(-1:1))
summary(tuned)