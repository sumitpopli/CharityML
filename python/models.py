import datetime as dt
from sklearn.metrics import accuracy_score, f1_score,fbeta_score,precision_score,recall_score,confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.base import clone
import numpy as np
import visuals as vs

#classifiers
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.ensemble import AdaBoostClassifier as abc

'''
Calculating accuracy and F-score model that always predicted an individual made more than $50,000, 
'''
def naive_predictor_performance(income):
    TP = np.sum(income)  # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data
    # encoded to numerical values done in the data preprocessing step.
    FP = income.shape[0] - TP  # Specific to the naive case

    TN = 0  # No predicted negatives in the naive case
    FN = 0  # No predicted negatives in the naive case

    # TODO: Calculate accuracy, precision and recall
    accuracy = TP / income.shape[0]
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    f1score = (precision * recall) / (precision + recall)
    fscore = ((1 + 0.5 * 0.5) * (precision * recall)) / ((0.5 * 0.5 * precision) + recall)

    # Print the results
    print("Naive Predictor: [Accuracy score: {0}, F-score: {1}]".format(accuracy, fscore))
    return accuracy, fscore

'''
Fitting and predicting ML algorithms
Calculating Fscore, accuracy of teh ML object passed 
'''
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    # X_sample_train = X_train.iloc[:sample_size,:]
    # y_sample_train = y_train[:sample_size]
    start = dt.datetime.now()  # Get start time
    learner = learner.fit(X_train.iloc[:sample_size, :], y_train[:sample_size])
    end = dt.datetime.now()  # Get end time
    delta_dt = (end - start).total_seconds()

    # Calculate the training time
    results['train_time'] = delta_dt

    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = dt.datetime.now()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train.iloc[:sample_size, :])
    end = dt.datetime.now()  # Get end time
    delta_dt = (end - start).total_seconds()

    # Calculate the total prediction time
    results['pred_time'] = delta_dt

    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300].ravel(), predictions_train[:300])

    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300].ravel(), predictions_train[:300], beta=0.5)

    # Compute F-score on the test set which is y_test

    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    # Success
    print("{0} trained on {1} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results
'''
evaluating performance of three models: Decision tree, Random Forest , Ada boost.
performance parms: time taken, accuracy, fscore on different sizes of data.
'''
def initialize_models(X_train, y_train, X_test, y_test, accuracy, fscore):
    # TODO: Initialize the three models
    clf_A = dtc(random_state=13)
    clf_B = rfc(random_state=13)
    clf_C = abc(random_state=13)

    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    # HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
    # HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
    samples_100 = len(y_train)
    samples_10 = len(y_train) // 10
    samples_1 = len(y_train) // 100

    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    vs.evaluate(results, accuracy, fscore)
    return clf_C

'''
tuning adaboost algorithm to produce higher accuracy rates.
'''
def model_tunings_abc(X_train, y_train, X_test, y_test):

    # Initialize the classifier
    base_model = rfc()
    clf = abc(base_estimator=base_model, random_state=13)

    # TODO: Create the parameters list you wish to tune, using a dictionary if needed.
    # HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
    parameters = {'learning_rate': [0.02, 0.04, 0.2], 'n_estimators': [75, 100, 150]}

    # TODO: Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score, beta=0.5)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train, y_train.ravel())

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    # Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train.ravel())).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    # print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("Accuracy score on testing data: {0}".format(accuracy_score(y_test, predictions)))
    # print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
    print("F-score on testing data: {0}".format(fbeta_score(y_test, predictions, beta=0.5)))
    print("\nOptimized Model\n------")
    # print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final accuracy score on the testing data: {0}".format(accuracy_score(y_test, best_predictions)))
    # print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
    print("Final F-score on the testing data: {0}".format(fbeta_score(y_test, best_predictions, beta=0.5)))

    return best_clf,best_predictions


'''
Comparing performance of classification algorithm with a reduced set of features based on 
Adaboost best_params attribute and the tuned adaboost classifier 
'''
def model_feature_selection(clf_c,best_clf, best_predictions,X_train, y_train, X_test, y_test):
    # TODO: Train the supervised model on the training set using .fit(X_train, y_train)
    model = None

    # TODO: Extract the feature importances using .feature_importances_
    importances = clf_c.feature_importances_

    # Plot
    vs.feature_plot(importances, X_train, y_train)
    # Reduce the feature space
    X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
    X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

    # Train on the "best" model found from grid search earlier
    fit_start = dt.datetime.now()
    clf = (clone(best_clf)).fit(X_train_reduced, y_train)
    fit_end = dt.datetime.now()

    fit_time = fit_end - fit_start

    # Make new predictions
    pred_start = dt.datetime.now()
    reduced_predictions = clf.predict(X_test_reduced)
    pred_end = dt.datetime.now()
    pred_time = pred_end - pred_start

    # Report scores from the final model using both versions of data
    print("Final Model trained on full data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
    print("\nFinal Model trained on reduced data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta=0.5)))
    print('time taken for training is {0}'.format(fit_time))
    print('time taken for predicting is {0}'.format(pred_time))
