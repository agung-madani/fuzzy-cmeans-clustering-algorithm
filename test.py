from fcm import fuzzycmeans
import pandas as pd

# Datasets taken from https://www.kaggle.com/datasets/rakeshrau/social-network-ads

data = pd.read_csv('Social_Network_Ads.csv')
data.head(10)

global df
df = data
# Replace Female & Male with 0 & 1
df['Gender'].replace(['Female', 'Male'], [0,1], inplace=True)
# Drop unnessecary columns
df = df.drop(['User ID'], axis=1)
# Replace NaN values with 0
df.replace(float('nan'), 0, inplace=True)

# Change columns name into number
columnnames = {}
count=0
for i in df.columns:
    columnnames[i] = count
    count += 1
df.rename(columns = columnnames, inplace = True)

fuzzycmeans(df,df,3,2,100,0.00001,0,1)
