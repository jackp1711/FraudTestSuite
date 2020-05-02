from ML_Models import LogisticRegression, NaiveBayes, NeuralNetwork, RandomForest, SVM
import sklearn.model_selection as model_selection
from Results import graphing, file_writer
from Results.run_result import Result
import pandas as pd
import numpy as np
from sklearn import preprocessing
from PreProcessing import DeepFeatureSynthesis as dfs


class AnalysisTool:

    data = None
    data_type = ''
    class_column = ''
    output_file = ''

    le = None

    models = []

    labels = []
    accuracy = []

    total_fraud = 0
    false_pos = []
    false_neg = []
    correct_fraud = []

    final_result = None

    def __init__(self, data, data_type, class_column):
        self.data = data
        self.le = preprocessing.LabelEncoder()
        self.data_type = data_type
        self.class_column = class_column

    def run_experiments(self, iterations, tts, rand, feature_engineering):

        results = [None] * iterations

        if feature_engineering == 1:
            feature_engineered = dfs.PreProcessor(self.data)
            self.data = feature_engineered.feature_matrix

        for i in range(iterations):
            run = self.run_test(tts, rand)
            # print(run.labels)
            results[i] = run

        # self.final_result = results[0]
        # print(results[0].labels)
        self.average_results(results)

    def average_results(self, results):
        # print(results)

        # print(results[1].labels)

        ave_result = Result()
        # print(ave_result.labels)
        ave_result.labels = results[0].labels
        ave_result.false_pos.append(0)
        ave_result.false_neg.append(0)
        # ave_result.correct_fraud.append(0)

        for i in range(len(results[0].labels)):
            ave_acc = 0
            ave_false_pos = 0
            ave_false_neg = 0
            ave_correct = 0

            for j in range(len(results)):
                total = results[j].total_fraud
                ave_acc += results[j].accuracy[i]
                false_neg = results[j].false_neg[i+1]
                false_pos = results[j].false_pos[i+1]

                ave_false_neg += false_neg/total
                ave_false_pos += false_pos/total
                ave_correct += results[j].correct_fraud[i] / total

            ave_acc = ave_acc / len(results)
            ave_false_pos = ave_false_pos / len(results)
            ave_false_neg = ave_false_neg / len(results)
            ave_correct = ave_correct / len(results)

            ave_result.accuracy.append(round(ave_acc, 3))
            ave_result.false_pos.append(round(ave_false_pos, 3))
            ave_result.false_neg.append(round(ave_false_neg, 3))
            ave_result.correct_fraud.append(round(ave_correct, 3))

        # print(ave_result.false_pos)
        # print(ave_result.false_neg)
        # print(ave_result.correct_fraud)
        # print("accuray results = " + str(ave_result.accuracy))
        # print("labels = " + str(ave_result.labels))
        self.final_result = ave_result

    def run_test(self, train_test_split, rand_state):

        if self.data_type == 'sim':
            for column_name in self.data.columns:
                if self.data[column_name].dtype == object:
                    self.data[column_name] = self.le.fit_transform(self.data[column_name])
                else:
                    pass

        X = self.data.loc[:, self.data.columns != self.class_column]
        y = self.data.loc[:, self.data.columns == self.class_column]
        y = np.ravel(y)

        # print(np.count_nonzero(y == 1))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=train_test_split, random_state=rand_state)

        # print(np.count_nonzero(y_train == 1))
        # print(np.count_nonzero(y_test == 1))

        result = Result()

        print("Total instances of fraud = ", result.total_fraud)
        result.total_fraud = np.count_nonzero(y_test == 1)
        result.false_pos.append(np.count_nonzero(y_test == 1))
        result.false_neg.append(0)
        result.correct_fraud.append(0)

        log_model = LogisticRegression.LogisticReg(X_train, y_train)
        log_model.fit_model()
        result.labels.append("Log Reg")
        result.accuracy.append(float("{0:.4f}".format(log_model.test(X_test, y_test))))
        # print("False Positives = ", log_model.false_positives)
        # print("False Negatives = ", log_model.false_negatives)
        result.false_pos.append(log_model.false_positives)
        result.false_neg.append(log_model.false_negatives)
        result.correct_fraud.append(log_model.correct_fraud_prediction)

        svm_model = SVM.SVMAnalysis(X_train, y_train)
        svm_model.fit_model()
        result.labels.append("SVM")
        result.accuracy.append(float("{0:.4f}".format(svm_model.test(X_test, y_test))))
        # print("False Positives = ", svm_model.false_positives)
        # print("False Negatives = ", svm_model.false_negatives)
        result.false_pos.append(svm_model.false_positives)
        result.false_neg.append(svm_model.false_negatives)
        result.correct_fraud.append(svm_model.correct_fraud_prediction)

        naive_bayes = NaiveBayes.NaiveBayes(X_train, y_train)
        naive_bayes.fit_model()
        result.labels.append("Naive Bayes")
        result.accuracy.append(float("{0:.4f}".format(naive_bayes.test(X_test, y_test))))
        # print("False Positives = ", naive_bayes.false_positives)
        # print("False Negatives = ", naive_bayes.false_negatives)
        result.false_pos.append(0)
        result.false_neg.append(naive_bayes.false_negatives)
        result.correct_fraud.append(naive_bayes.correct_fraud_prediction)

        neural_net = NeuralNetwork.NeuralNet(X_train, y_train)
        neural_net.fit_model()
        result.labels.append("Neural Net")
        result.accuracy.append(float("{0:.4f}".format(neural_net.test(X_test, y_test))))
        # print("False Positives = ", neural_net.false_positives)
        # print("False Negatives = ", neural_net.false_negatives)
        result.false_pos.append(neural_net.false_positives)
        result.false_neg.append(neural_net.false_negatives)
        result.correct_fraud.append(neural_net.correct_fraud_prediction)

        rand_forest = RandomForest.RandomForest(X_train, y_train)
        rand_forest.fit_model()
        result.labels.append("Random forest")
        result.accuracy.append(float("{0:.4f}".format(rand_forest.test(X_test, y_test))))
        # print("False Positives = ", rand_forest.false_positives)
        # print("False Negatives = ", rand_forest.false_negatives)
        result.false_pos.append(rand_forest.false_positives)
        result.false_neg.append(rand_forest.false_negatives)
        result.correct_fraud.append(rand_forest.correct_fraud_prediction)

        return result

    def print_to_console(self):
        print("Total instances of fraud = " + str(self.total_fraud))

        print("False positives for logistic regression = " + str(self.final_result.false_pos[0]))
        print("False negatives for logistic regression = " + str(self.final_result.false_neg[0]))
        print("Correct fraud predictions = " + str(self.final_result.correct_fraud[0]))

        print("False SVM Positives = " + str(self.final_result.false_pos[1]))
        print("False SVM negatives = " + str(self.final_result.false_neg[1]))
        print("Correct fraud predictions = " + str(self.final_result.correct_fraud[1]))

        print("False Naive Bayes Positives = " + str(self.final_result.false_pos[2]))
        print("False Naive Bayes negatives = " + str(self.final_result.false_neg[2]))
        print("Correct fraud predictions = " + str(self.final_result.correct_fraud[2]))

        print("False Neural Network Positives = " + str(self.final_result.false_pos[3]))
        print("False Neural Network negatives = " + str(self.final_result.false_neg[3]))
        print("Correct fraud predictions = " + str(self.final_result.correct_fraud[3]))

        print("False Random Forest Positives = " + str(self.final_result.false_pos[4]))
        print("False Random Forest negatives = " + str(self.final_result.false_neg[4]))
        print("Correct fraud predictions = " + str(self.final_result.correct_fraud[4]))

    def write_results_to_file(self, o_file):
        self.output_file = o_file

        file = file_writer.File_writer(self.output_file)

        file.write_file("Total instances of fraud = " + str(self.total_fraud))

        file.write_file("False positives for logistic regression = " + str(self.final_result.false_pos[0]))
        file.write_file("False negatives for logistic regression = " + str(self.final_result.false_neg[0]))
        file.write_file("Correct fraud predictions = " + str(self.final_result.correct_fraud[0]))

        file.write_file("False SVM Positives = " + str(self.final_result.false_pos[1]))
        file.write_file("False SVM negatives = " + str(self.final_result.false_neg[1]))
        file.write_file("Correct fraud predictions = " + str(self.final_result.correct_fraud[1]))

        file.write_file("False Naive Bayes Positives = " + str(self.final_result.false_pos[2]))
        file.write_file("False Naive Bayes negatives = " + str(self.final_result.false_neg[2]))
        file.write_file("Correct fraud predictions = " + str(self.final_result.correct_fraud[2]))

        file.write_file("False Neural Network Positives = " + str(self.final_result.false_pos[3]))
        file.write_file("False Neural Network negatives = " + str(self.final_result.false_neg[3]))
        file.write_file("Correct fraud predictions = " + str(self.final_result.correct_fraud[3]))

        file.write_file("False Random Forest Positives = " + str(self.final_result.false_pos[4]))
        file.write_file("False Random Forest negatives = " + str(self.final_result.false_neg[4]))
        file.write_file("Correct fraud predictions = " + str(self.final_result.correct_fraud[4]))

    def graph_results(self, title_data):
        graphing_acc = graphing.Plot(self.final_result.accuracy, np.zeros(len(self.final_result.accuracy)),
                                     self.final_result.labels)
        acc_title = title_data + 'accuracy scores'
        graphing_acc.plot_bar_chart_comparison_acc_score(acc_title)

        self.final_result.labels.insert(0, "Total fraud")
        graphing_false = graphing.Plot(self.final_result.false_pos, self.final_result.false_neg,
                                       self.final_result.labels)
        comparison_title = title_data + 'incorrect results comparison'
        graphing_false.plot_bar_chart_comparison_false(self.final_result.correct_fraud, comparison_title)
