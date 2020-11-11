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

def calc_cros_entropy(m,p, y):
	epsilon = 1e-5    # fix to log problem
	out = 0.0
	for i in range(m):
		out += y[i] * np.log(p[i] + epsilon) + (1.0 - y[i]) * np.log(1.0 - p[i] + epsilon)
	return (-1 /m * out)