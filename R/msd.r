library (e1071)
SongCSV = read.csv('./SongCSV3.csv')
# allRows = 1:nrow(SongCSV)
# testRows = sample(allRows, trunc(length(allRows) * 0.3))
# SongCSV_test = SongCSV[testRows,]
# SongCSV_train = SongCSV[-testRows,]
# model = svm(letter~., data = SongCSV_train, kernel="radial", gamma = .1, cost = 100, cross = 10)
# prediction = predict(model, SongCSV_test[,-1])
# confusion_matrix = table(pred = prediction, true = SongCSV_test$letter)
# agreement = prediction == SongCSV_test$letter
# accuracy = prop.table(table(agreement))
#
# #print(confusion_matrix)
# print(accuracy)
# #plot(model, SongCSV)
# tuned <- tune.svm(letter~., data = SongCSV_train, gamma = 10^(-6:-1), cost = 10^(-1:1))
# summary(tuned)
