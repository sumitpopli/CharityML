
import pandas as pd
import numpy as np

import matplotlib as mtp
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import visuals as vs
import models as m

'''
Splitting data in to testing and training data for ML algorithms
'''
def split_data(final_features, income):
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(final_features,
                                                        income,
                                                        test_size=0.2,
                                                        random_state=0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test

'''
cleaning, formating, and restructuring data certain features that must be adjusted. 
'''
def data_preprocessing(data):
    # Split the data into features and target label
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    # Visualize skewed continuous features of original data
    vs.distribution(data)

    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    # Visualize the new log distributions
    vs.distribution(features_log_transformed, transformed=True)
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()  # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # Show an example of a record with scaling applied
    print(features_log_minmax_transform.head(n=5))

    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_log_minmax_transform,
                                    prefix=['workclass', 'education_level', 'marital-status', 'occupation',
                                            'relationship', 'race', 'sex', 'native-country'],
                                    columns=['workclass', 'education_level', 'marital-status', 'occupation',
                                             'relationship', 'race', 'sex', 'native-country'])

    # Encode the 'income_raw' data to numerical values
    encoder = LabelEncoder()
    income = encoder.fit_transform(income_raw)

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))

    # Uncomment the following line to see the encoded feature names
    print(encoded)
    print(income)

    return features_final, income

'''
Explore the dataset to determine how many individuals 
fit into either >50K or < 50K group, and print the percentage of these individuals making 
more than $50,000
'''
def data_exploration(data):
    print(data.head(n=1))
    #Total number of records
    n_records = data.shape[0]

    #Number of records where individual's income is more than $50,000
    n_greater_50k = data.loc[data['income'] == '>50K'].shape[0]

    #Number of records where individual's income is at most $50,000
    n_at_most_50k = data.loc[data['income'] == '<=50K'].shape[0]

    #Percentage of individuals whose income is more than $50,000
    greater_percent = (n_greater_50k * 100) / n_records

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


#Main function: Load the CSV file
if __name__ == "__main__":
    #read the csv file in to pandas
    people_data = pd.read_csv('census.csv')

    #explore data to get an idea
    data_exploration(people_data)

    # seperating the X and y features. applying  scaling, encoding etc to teh data
    final_features, income = data_preprocessing(people_data)

    #splitting data in to train and test.
    X_train, X_test, y_train, y_test = split_data(final_features, income)
    print('X_train shape {0}, x_test shape {1}, y_train shape {2}, y_test shape {3}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    #getting accuracy score and fscore if the model predicted that all individuals earned more than 50K
    accuracy, fscore = m.naive_predictor_performance(income)

    # use 3 different models to calculate the performance.
    clf_untuned_adaboost = m.initialize_models(X_train,  y_train,X_test, y_test,accuracy, fscore)

    #tuning adaboost model for max performance.
    clf_tuned_adaboost, predictions = m.model_tunings_abc(X_train,  y_train,X_test, y_test)


    # compare performance of tuned model vs model with best params but un tuned.
    m.model_feature_selection(clf_untuned_adaboost,clf_tuned_adaboost, predictions, X_train,  y_train,X_test, y_test)

