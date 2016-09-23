class Hard_Coded:

    def __init__(self):
        self.data = []
        self.targets = []

    def train(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, data, targets):
        predictions = []
        correct_prediction = 0.0
        for i in range(len(targets)):
            predicted_number = 1
            predictions.append(predicted_number)

            if predictions[i] == targets[i]:
                correct_prediction += 1

        predict_percent = ((correct_prediction/len(targets)*100))
        return predict_percent
