import pandas as pd

train = pd.read_csv('train.csv')

print(train.head())

trainSubset = train.iloc[1:100000]
trainSubset.to_csv('train_tmp.csv')

testSubset = train.iloc[100001:120000]
testSubset.to_csv('test.csv')