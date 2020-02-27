from sklearn.neural_network import MLPClassifier


class NeuralNet:

    X = []
    y = []

    model = None

    predictions = None
    accuracy = None

    correct_predictions = 0
    false_positives = 0
    false_negatives = 0
    correct_fraud_prediction = 0

    def __init__(self, x, y):
        self.X = x
        self.y = y
        return

    def fit_model(self):
        self.model = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=500, alpha=0.0001, solver='sgd',
                                   random_state=21, tol=0.000000001)
        self.model.fit(self.X, self.y)

        return self.model

    def predict(self, test_data):
        self.predictions = self.model.predict(test_data)

        return self.predictions

    def test(self, x_test, y_real):
        self.correct_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0

        self.predict(x_test)

        for i in range(len(self.predictions)):
            if self.predictions[i] == y_real[i]:
                self.correct_predictions += 1
                if self.predictions[i] == 1:
                    self.correct_fraud_prediction += 1

            elif self.predictions[i] == 0 and y_real[i] == 1:
                self.false_negatives += 1

            elif self.predictions[i] == 1 and y_real[i] == 0:
                self.false_positives += 1

        score = self.correct_predictions / len(x_test)

        return score
