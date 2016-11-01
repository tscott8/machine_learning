library (e1071)
vowel = read.csv('../datasets/vowel.csv', header = TRUE)
allRows = 1:nrow(vowel)
testRows = sample(allRows, trunc(length(allRows) * 0.3))
vowel_test = vowel[testRows,]
vowel_train = vowel[-testRows,]
model = svm(Class~., data = vowel_train, kernel="radial", gamma = .1, cost = 100, cross = 10)
prediction = predict(model, vowel_test[,-13])
confusion_matrix = table(pred = prediction, true = vowel_test$Class)
agreement = prediction == vowel_test$Class
accuracy = prop.table(table(agreement))

print(confusion_matrix)
print(accuracy)
#tuned <- tune.svm(Class~., data = vowel_train, gamma = 10^(-6:-1), cost = 10^(-1:1))
#summary(tuned)