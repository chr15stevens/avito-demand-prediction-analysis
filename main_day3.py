import math
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

train = pd.read_csv('train_tmp.csv').fillna(0)
test = pd.read_csv('test.csv').fillna(0)

print(train.head())
print(train.axes)

x_train = train[['price', 'item_seq_number', 'image_top_1']]
x_test = test[['price', 'item_seq_number', 'image_top_1']]
y_train = train.deal_probability
y_test = test.deal_probability

lrFull = linear_model.LinearRegression()
lrFull.fit(x_train, y_train)
predictions_full = lrFull.predict(x_test)

# mean squared error on full dataset logistic regression
meanSquaredError = np.sum(np.square(y_test - predictions_full))/predictions_full.size
rootMeanSquaredError = math.sqrt(meanSquaredError)
print('Full dataset root mean squared error: ', rootMeanSquaredError)

groupableColumnLabels = ['region', 'city', 'parent_category_name', 'category_name', 'user_type']

groupModelsDict = {}

for groupableColumnLabel in groupableColumnLabels:
    print('Building regression models ' + groupableColumnLabel)
    # split our dataframe into a set of dataframes for each parent_category
    train_grouping = train.groupby(groupableColumnLabel)
    train_groups_dict = {}
    [train_groups_dict.__setitem__(x,train_grouping.get_group(x)) for x in train_grouping.groups]

    # build our set of regression models one for each parent_category_name
    regression_models_dict = {}
    for key, train_group in train_groups_dict.items():
        lr = linear_model.LinearRegression()
        lr.fit(train_group[['price', 'item_seq_number', 'image_top_1']], train_group.deal_probability)
        regression_models_dict[key] = lr

    groupModelsDict[groupableColumnLabel] = regression_models_dict

# iterate over all rows in our test data and build a new row of predictions, one for each category column
print('Making predictions')
predictions = []
for index, row in test.iterrows():
    row_data = np.reshape([row['price'], row['item_seq_number'], row['image_top_1']], (1,-1))
    
    rowPredictions = []
    for groupableColumnLabel in groupableColumnLabels:
        groupableColumnValue = row[groupableColumnLabel]
        # if a model is missing for whatever reason pick up the full model to at least get a number
        if groupableColumnValue in regression_models_dict:
            prediction = groupModelsDict[groupableColumnLabel][groupableColumnValue].predict(row_data)
        else:
            prediction = lrFull.predict(row_data)
        rowPredictions.append(min(max(0,prediction[0]),1))

    predictions.append(rowPredictions)

meanPredictions = [sum(p)/5 for p in predictions]

# mean squared error on individual parent_category logistic regression
meanSquaredError = np.sum(np.square(y_test - meanPredictions))/len(meanPredictions)
rootMeanSquaredError = math.sqrt(meanSquaredError)
print('Mean predicition root mean squared error: ', rootMeanSquaredError)

