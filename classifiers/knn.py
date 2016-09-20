class  k_Nearest_Neighbor:

    def train(self, dataset):
        return

    def predict(self, dataset):
        predictions = []
        correct_prediction = 0.0
        for i, data_point in enumerate(dataset):
            predicted_number = 1
            predictions.append(predicted_number)

            if predictions[i] == data_point:
                correct_prediction += 1

        predict_percent = ((correct_prediction/len(dataset)*100))
        print("Method Percentage = " + str(predict_percent) + "%")
