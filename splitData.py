import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

features = pd.read_csv("features.csv")

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(features, features['activity']):
	train_set = features.loc[train_index]
	test_set = features.loc[test_index]

test_set = pd.DataFrame(data = test_set)
#test_set.drop(test_set.columns[0], axis = 1)
print(test_set)


split2 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 42)
for training_index, validation_index in split2.split(train_set, train_set['activity']):
	training_set = train_set.loc[training_index]
	validation_set = train.loc[validation_index]


print("features: len = " + str(len(features)) + "atributes: " + str(len(features ['activity'].unique())) + "")
print("train_set: len = " + str(len(train_set)) + "atributes: " + str(len(train_set['activity'].unique())) + "")
print("test_set: len = " + str(len(test_set)) + "atributes: " + str(len(test_set['activity'].unique())) + "")
print("training_set: len = " + str(len(training_set)) + "atributes: " + str(len(training_set['activity'].unique())) + "")
print("validation_set: len = " + str(len(validation_set)) + "atributes: " + str(len(validation_set['activity'].unique())) + "")


"""
test_set.to_csv("test_set.csv")
training_set.to_csv("train_set.csv")
validation_set.to_csv("validation_set.csv")
"""
