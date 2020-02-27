from Analysis import LogisticRegression, NaiveBayes, NeuralNetwork, RandomForest, SVM
import sklearn.model_selection as model_selection
from Results import graphing, file_writer
import pandas as pd
import numpy as np
from sklearn import preprocessing

chunk_list = []

chunked_data = pd.read_csv('C:\\Users\\jacky\\PycharmProjects\\FraudTestSuite\\data\\simulated_data.csv',
                           chunksize=100000, low_memory=False, dtype={'step': 'int', 'type': object, 'amount': 'float',
                            'nameOrig': object, 'oldBalanceOrig': 'float', 'newBalanceOrig': 'float', 'nameDest': object
                            , 'oldBalanceDest': 'float', 'newBalanceDest': 'float', 'isFraud': 'int', 'isFlaggedFraud':
                            'int'})

for chunk in chunked_data:
    chunk_list.append(chunk)

# simulated_data = pd.concat(chunk_list)
new_list = [chunk_list[0], chunk_list[1], chunk_list[2], chunk_list[3]]
simulated_data = pd.concat(new_list)

models = []

labels = []
accuracy = []

false_pos = []
false_neg = []
correct_fraud = []

le = preprocessing.LabelEncoder()

for column_name in simulated_data.columns:
    if simulated_data[column_name].dtype == object:
        simulated_data[column_name] = le.fit_transform(simulated_data[column_name])
    else:
        pass

X = simulated_data.loc[:, simulated_data.columns != 'isFraud']
y = simulated_data.loc[:, simulated_data.columns == 'isFraud']
y = np.ravel(y)


print(np.count_nonzero(y == 1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.80, random_state=101)

print(np.count_nonzero(y_train == 1))
print(np.count_nonzero(y_test == 1))

file = file_writer.File_writer()
print("Total instances of fraud = ", np.count_nonzero(y_test == 1))
file.write_file("Total instances of fraud = " + str(np.count_nonzero(y_test == 1)))
false_pos.append(np.count_nonzero(y_test == 1))
false_neg.append(0)
correct_fraud.append(0)

log_model = LogisticRegression.LogisticReg(X_train, y_train)
log_model.fit_model()
labels.append("Log Reg")
accuracy.append(float("{0:.4f}".format(log_model.test(X_test, y_test))))
print("False Positives = ", log_model.false_positives)
print("False Negatives = ", log_model.false_negatives)
false_pos.append(log_model.false_positives)
false_neg.append(log_model.false_negatives)
correct_fraud.append(log_model.correct_fraud_prediction)
print("False Negatives = ", log_model.false_negatives)
file.write_file("False logistic regression Positives = " + str(log_model.false_positives))
file.write_file("False logistic regression negatives = " + str(log_model.false_negatives))
file.write_file("Correct fraud predictions = " + str(log_model.correct_fraud_prediction))

svm_model = SVM.SVMAnalysis(X_train, y_train)
svm_model.fit_model()
labels.append("SVM")
accuracy.append(float("{0:.4f}".format(svm_model.test(X_test, y_test))))
print("False Positives = ", svm_model.false_positives)
print("False Negatives = ", svm_model.false_negatives)
false_pos.append(svm_model.false_positives)
false_neg.append(svm_model.false_negatives)
correct_fraud.append(svm_model.correct_fraud_prediction)
file.write_file("False SVM Positives = " + str(svm_model.false_positives))
file.write_file("False SVM negatives = " + str(svm_model.false_negatives))
file.write_file("Correct fraud predictions = " + str(svm_model.correct_fraud_prediction))

naive_bayes = NaiveBayes.NaiveBayes(X_train, y_train)
naive_bayes.fit_model()
labels.append("Naive Bayes")
accuracy.append(float("{0:.4f}".format(naive_bayes.test(X_test, y_test))))
print("False Positives = ", naive_bayes.false_positives)
print("False Negatives = ", naive_bayes.false_negatives)
false_pos.append(0)
false_neg.append(naive_bayes.false_negatives)
correct_fraud.append(naive_bayes.correct_fraud_prediction)
file.write_file("False Naive Bayes Positives = " + str(naive_bayes.false_positives))
file.write_file("False Naive Bayes negatives = " + str(naive_bayes.false_negatives))
file.write_file("Correct fraud predictions = " + str(naive_bayes.correct_fraud_prediction))

neural_net = NeuralNetwork.NeuralNet(X_train, y_train)
neural_net.fit_model()
labels.append("Neural Net")
accuracy.append(float("{0:.4f}".format(neural_net.test(X_test, y_test))))
print("False Positives = ", neural_net.false_positives)
print("False Negatives = ", neural_net.false_negatives)
false_pos.append(neural_net.false_positives)
false_neg.append(neural_net.false_negatives)
correct_fraud.append(neural_net.correct_fraud_prediction)
file.write_file("False Neural Network Positives = " + str(neural_net.false_positives))
file.write_file("False Neural Network negatives = " + str(neural_net.false_negatives))
file.write_file("Correct fraud predictions = " + str(naive_bayes.correct_fraud_prediction))

rand_forest = RandomForest.RandomForest(X_train, y_train)
rand_forest.fit_model()
labels.append("Random forest")
accuracy.append(float("{0:.4f}".format(rand_forest.test(X_test, y_test))))
print("False Positives = ", rand_forest.false_positives)
print("False Negatives = ", rand_forest.false_negatives)
false_pos.append(rand_forest.false_positives)
false_neg.append(rand_forest.false_negatives)
correct_fraud.append(rand_forest.correct_fraud_prediction)
file.write_file("False Random Forest Positives = " + str(rand_forest.false_positives))
file.write_file("False Random Forest negatives = " + str(rand_forest.false_negatives))
file.write_file("Correct fraud predictions = " + str(rand_forest.correct_fraud_prediction))

graphing_acc = graphing.Plot(accuracy, np.zeros(len(accuracy)), labels)
graphing_acc.plot_bar_chart_comparison_acc_score()

labels.insert(0, "Total fraud")
graphing_false = graphing.Plot(false_pos, false_neg, labels)
graphing_false.plot_bar_chart_comparison_false(correct_fraud)
