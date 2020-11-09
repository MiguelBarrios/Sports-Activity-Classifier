from scipy.stats import skew
import os
import csv
import pandas as pd
from tqdm import tqdm

def extractFeatures(data):
	mean = data.mean()
	rmax = data.max()
	rmin = data.min()
	variance = data.var()
	std = data.std()
	skew1  = skew(data)
	features = [mean, rmax, rmin, variance, std, skew1]
	return features

def makeFeatureHeader(features, colNames):
	header = []
	for col in colNames:
		for feature in features:
			header.append(col + "_" + feature)
	header.append("activity")
	return header


features = ["mean", "max", "min", "var", "std", "skew"]
activites = {'a01': 'sitting', 'a02': 'standing', 'a03': 'lyingBack','a04':'lyingRigh','a05':'ascendingStairs','a06':'decendingStairs', 'a07':'standingInElevatorStill','a08':'movingInElevator','a09':'walkingLot','a10':'walkingTreadmillFlat', 'a11':'walkingTreadmillIncline','a12':'runningTreadmill','a13':'stepper', 'a14':'crossTrainer', 'a15':'cyclingHorizontal','a16':'cyclingVertical','a17':'rowing','a18':'jumping','a19':'basketBall'} 
people = ['p1','p2','p3','p4','p5','p6','p7','p8']

collumNames = ["T_xacc", "T_yacc", "T_zacc", "T_xgyro","T_ygyro","T_zgyro","T_xmag", "T_ymag", "T_zmag",
"RA_xacc", "RA_yacc", "RA_zacc", "RA_xgyro", "RA_ygyro","RA_zgyro", "RA_xmag", "RA_ymag", "RA_zmag",
"LA_xacc", "LA_yacc", "LA_zacc", "LA_xgyro", "LA_ygyro","LA_zgyro", "LA_xmag", "LA_ymag", "LA_zmag",
"RL_xacc", "RL_yacc", "RL_zacc", "RL_xgyro", "RL_ygyro","RL_zgyro", "RL_xmag", "RL_ymag", "RL_zmag",
"LL_xacc", "LL_yacc", "LL_zacc", "LL_xgyro", "LL_ygyro","LL_zgyro", "LL_xmag", "LL_ymag", "LL_zmag"]

mainDir = "data/"
features_df = []
header = makeFeatureHeader(features, collumNames)
features_df.append(header)
for activity in tqdm(activites):
	for person in tqdm(people):
		#print(person + " " + activites[activity])
		segments = os.listdir(mainDir + activity + "/" + person)
		segments.sort()
		if ".DS_Store" in segments:
			segments.remove(".DS_Store")
		for segment in segments:
			row = []
			fpath = mainDir + activity + "/" + person + "/" + segment
			df = pd.read_csv(fpath, names = collumNames)
			for col in collumNames:
				features = extractFeatures(df[col])
				row.extend(features)
			row.append(activites[activity])
			features_df.append(row)

with open("features.csv", "w", newline = "") as f:
	writer = csv.writer(f)
	writer.writerows(features_df)
















