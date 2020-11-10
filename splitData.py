import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("features.csv")

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(df, df['activity']):
	train_set = df.loc[train_index]
	test_set = df.loc[test_index]

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 42)
for train_index, validation_index in split.split(train_set, train_set['activity']):
	training_set = df.loc[train_index]
	validation_set = df.loc[validation_index]

test_set.to_csv("test_set.csv")
training_set.to_csv("train_set.csv")
validation_set.to_csv("validation_set.csv")
