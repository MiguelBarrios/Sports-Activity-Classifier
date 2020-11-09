import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

df = pd.read_csv("features.csv")
features = df.columns[0:-1]
target = 'activity'
pca_df = principal_component_analysis(df = df, features = features,target =  target, n = 30)

