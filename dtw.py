import pandas as pd
import numpy as np 
import pyarrow as py
from tslearn.clustering import TimeSeriesKMeans , silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import time

start_time = time.time()

df = pd.read_parquet('/dados/samuelrt/NewExperimeto/data/data_2023-10-17_12_22_16.parquet', engine='pyarrow')
df.set_index(pd.to_datetime(df['datetime']), inplace=True)
df.drop('datetime', axis=1, inplace=True)
start_date = '2021-01-01'
end_date = '2021-12-31'

var = ['latitude', 'longitude', 'precipitation', 'humidity', 'temperature', 'wind_speed']

estation = []  # store names from each station
tensor = []    # store lat an long from each station
#values = {} # store lat long and phtw
pos = {}

for station in df['station'].unique():
    frame = df.loc[df['station'] == station].loc[:, var]
    frame = frame.sort_index().loc[start_date:end_date]
    lat_long = tuple(frame[['latitude', 'longitude']].drop_duplicates().values.flatten())
    frame = frame.loc[:, var[2:]]
    frame = frame.resample(rule='D').sum().values
    if (frame.shape[0] == 365) and (np.isnan(frame).sum() == 0):
        estation.append(station)
        tensor.append(frame)
        pos[station] = lat_long

tensor = np.array(tensor)
scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(tensor)
scores_dtw = []

for k in range(3,11):
    km_dtw = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=25,
                          max_iter_barycenter=50, n_jobs=24,
                          random_state=0)
    labels = km_dtw.fit_predict(X)
    scores_dtw.append((silhouette_score(X, labels, metric="dtw",n_jobs=24),labels,k))

max_dtw = max(scores_dtw, key=lambda x: x[0])

data = pd.DataFrame()
data.loc[:,'station'] = estation
data.loc[:,f'D{max_dtw[2]}'] = max_dtw[1]
data.to_csv('dtw.csv',index=False)

silhouette = pd.DataFrame()
silhouette.loc[:,'k'] = range(3,11)
silhouette.loc[:,'score'] = scores_dtw
silhouette.to_csv('silueta.csv',index=False)

end_time = time.time()

print('time',end_time-start_time)
