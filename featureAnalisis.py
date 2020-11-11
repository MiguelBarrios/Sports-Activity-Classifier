from functions import * 

df = pd.read_csv("train_set.csv")
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

