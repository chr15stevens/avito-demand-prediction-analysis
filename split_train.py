import pandas as pd

train = pd.read_csv('train.csv')

trainSubset = train.iloc[1:50000]
trainSubset.to_csv('train_tmp.csv')