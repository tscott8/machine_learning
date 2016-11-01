library (e1071)
abalone = read.csv('../datasets/abalone.csv', header = TRUE)
allRows = 1:nrow(abalone)
testRows = sample(allRows, trunc(length(allRows) * 0.3))
abalone_test = abalone[testRows,]
abalone_train = abalone[-testRows,]
model = svm(Rings~., data = abalone_train, kernel="radial", gamma = .1, cost = 1, cross = 10)
prediction = round(predict(model, abalone_test[,-9]))
confusion_matrix = table(pred = prediction, true = abalone_test$Rings)
agreement = prediction == abalone_test$Rings
accuracy = prop.table(table(agreement))

print(confusion_matrix)
print(accuracy)
