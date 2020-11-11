from functions import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

train_set = pd.read_csv("train_set.csv")
test_set = pd.read_csv("test_set.csv")

features = train_set.columns[0:-1]
target = 'activity'

pca_train = principal_component_analysis(df = train_set, features = features, target = target, n = 30)
pca_test = principal_component_analysis(df = test_set, features = features, target = target, n = 30)


activites = ['sitting','standing','lyingBack','lyingRigh',
'ascendingStairs','decendingStairs','standingInElevatorStill','movingInElevator',
'walkingLot','walkingTreadmillFlat','walkingTreadmillIncline','runningTreadmill',
'stepper','crossTrainer','cyclingHorizontal','cyclingVertical','rowing','jumping',
'basketBall']

replace = range(1,20)

for i in range(19):
	pca_train['activity'] = pca_train['activity'].replace([activites[i]], replace[i])
	pca_test['activity'] = pca_test['activity'].replace([activites[i]], replace[i])

X_train = pca_train.iloc[:,0:-1]
y_train = pca_train['activity'].values

# fit Decision tree
X_test = pca_test.iloc[:,0:-1]
y_test = pca_test['activity'].values

for i in range(1,40):
	print("Depth: " + str(i) + "  ", end = "")
	depth = i
	decision_tree = DecisionTreeClassifier(max_depth = depth, criterion = "entropy") #entropy
	decision_tree = decision_tree.fit(X_train, y_train)
	res_pred = decision_tree.predict(X_test)
	score = accuracy_score(y_test, res_pred)
	print(score)


"""
Optimial decision tree cur config: pca = 30, depth = 28, criterion = entropy, acc = 66%

"""





























