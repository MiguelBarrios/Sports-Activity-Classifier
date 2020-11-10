import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

def principal_component_analysis(df, features, target, n):
	#Standardize
	x = df.loc[:,features].values
	y = df.loc[:,target].values
	x = StandardScaler().fit_transform(x)
	components = []
	for i in range(1,n + 1):
		components.append("pc" + str(i))
	pca = PCA(n_components=n)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data = principalComponents, columns = components)
	return pd.concat([principalDf, df[[target]]], axis = 1)

def plot_pca(pca_df, targets, colors, pc1, pc2):
	fig = plt.figure(figsize = (12,12))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel(pc1, fontsize = 20)
	ax.set_ylabel(pc2, fontsize = 20)
	ax.set_title(pc1 + ' x ' + pc2, fontsize = 40)
	for target, color in zip(targets,colors):
	    indicesToKeep = pca_df['activity'] == target
	    ax.scatter(pca_df.loc[indicesToKeep, pc1]
	               , pca_df.loc[indicesToKeep, pc2]
	               , c = color
	               , s = 50)
	ax.legend(targets)
	ax.grid()
	plt.show()

df = pd.read_csv("features.csv")
features = df.columns[0:-1]
target = 'activity'
pca_df = principal_component_analysis(df = df, features = features,target =  target, n = 30)


########## Analisis    ##########
targets = ['sitting','standing','lyingBack','lyingRigh',
'ascendingStairs','decendingStairs','standingInElevatorStill','movingInElevator',
'walkingLot','walkingTreadmillFlat','walkingTreadmillIncline','runningTreadmill',
'stepper','crossTrainer','cyclingHorizontal','cyclingVertical','rowing','jumping',
'basketBall']


colors = ['black','black','black','black',
'r','r','r','r',
'limegreen','limegreen','limegreen','limegreen',
'b','darkolivegreen','b','b','orange','gold',
'indigo']
#colors =cm.rainbow(np.linspace(0,1,30))

pc1 = 'pc1'
pc2 = 'pc2'
#plot_pca(pca_df, targets, colors, pc1,pc2)
pcalist = pca_df.columns.values[0:-1]

for pca in pcalist:
	plot_pca(pca_df, targets, colors, pc1,pca)


