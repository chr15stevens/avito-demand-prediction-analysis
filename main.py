import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

train = pd.read_csv('train_tmp.csv')

print(train.head())
print(train.axes)

trainSubset = train[['price', 'deal_probability', 'item_seq_number']]

numberOfStandardDeviationsToInclude = 0.5

trainSubset[((trainSubset.price - trainSubset.price.mean()) / trainSubset.price.std()).abs() < numberOfStandardDeviationsToInclude]
trainSubset[((trainSubset.deal_probability - trainSubset.deal_probability.mean()) / trainSubset.deal_probability.std()).abs() < numberOfStandardDeviationsToInclude]
trainSubset[((trainSubset.item_seq_number - trainSubset.item_seq_number.mean()) / trainSubset.item_seq_number.std()).abs() < numberOfStandardDeviationsToInclude]

pd.plotting.scatter_matrix(trainSubset, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show()
print(trainSubset.head())