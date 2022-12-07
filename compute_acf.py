import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371000):
    # Convert degrees to radians
    phi1, lambda1, phi2, lambda2 = [c*np.pi/180 for c in (lat1, lon1, lat2, lon2)]      
    # Apply the haversine formula
    a = (np.square(np.sin((phi2-phi1)/2)) + np.cos(phi1) * np.cos(phi2) * 
         np.square(np.sin((lambda2-lambda1)/2)))
    d = 2*r*np.arcsin(np.sqrt(a))
    return d


# Data.tsv is stored locally in the
# same directory as of this python file
df = pd.read_csv('data/Roots_data_2021_2022.txt',sep = '\t')
df = df[df['days_sampled'].notnull()]
df = df[df['days_sampled'] > 0]

print(df.columns)
N = df.shape[0]
sensor_id = df['Sensor_ID'].values.reshape(N,1)

spores = df['spores'].values.reshape(N,1)

#Collection_end_date = df['Collection_end_date'].values
#Collection_start_date = df['Collection_start_date'].values

days_sampled = df['days_sampled'].values.reshape(N,1)
end_day = df['j_date'].values.reshape(N,1)
beg_day = end_day - days_sampled

center_date = end_day - 0.5 * days_sampled
days_offset = np.abs(center_date - center_date.T)
# do an intersect over union (IoU) for time overlap
time_overlap = (np.maximum(0, np.minimum(end_day, end_day.T) - np.maximum(beg_day, beg_day.T)) / \
    (np.maximum(end_day, end_day.T) - np.minimum(beg_day, beg_day.T))).flatten()

Lat = df['Latitude'].values.reshape(N,1)
Long = df['Longitude'].values.reshape(N,1)
DistMat = calculate_spherical_distance(Lat, Long, Lat.T, Long.T).flatten()

xy = (spores * spores.T).flatten()
x2 = (np.square(spores) * np.ones((1,N))).flatten()
y2 = (np.ones((N,1)) * np.square(spores.T)).flatten()

# do the time ACF first
same_sensor = np.where(((sensor_id == sensor_id.T) * (days_offset>0)).flatten())[0]
days_offset = days_offset.flatten()

time_bin_sz_days = 4
d_offset = (np.round(days_offset[same_sensor] / time_bin_sz_days) * time_bin_sz_days).astype(np.int32)
un_days, inds = np.unique(d_offset, return_inverse=True)

ACF_time = np.zeros(un_days.size)
Ndatapts = np.zeros(un_days.size)
for i in range(un_days.size):
    inds2 = np.where(inds == i)[0]
    ACF_time[i] = np.sum(xy[same_sensor[inds2]]) / np.sqrt(np.sum(x2[same_sensor[inds2]])*np.sum(y2[same_sensor[inds2]]))
    Ndatapts[i] = inds2.size

fig, ax = plt.subplots()
plt.plot(un_days,ACF_time)
ax2 = ax.twinx()
plt.plot(un_days, Ndatapts,'r')
plt.show()
d = 1






