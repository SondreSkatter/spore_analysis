import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt

use_log = True

def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371000):
    # Convert degrees to radians
    phi1, lambda1, phi2, lambda2 = [c*np.pi/180 for c in (lat1, lon1, lat2, lon2)]      
    # Apply the haversine formula
    a = (np.square(np.sin((phi2-phi1)/2)) + np.cos(phi1) * np.cos(phi2) * 
         np.square(np.sin((lambda2-lambda1)/2)))
    d = 2*r*np.arcsin(np.sqrt(a))
    return d

def date_to_day(dates):
    return np.array([d.timestamp()/24/3600 for d in dates])

# Data.tsv is stored locally in the
# same directory as of this python file
df = pd.read_csv('data/Roots_data_2021_2022.txt',sep = '\t')
df = df[df['days_sampled'].notnull()]
df = df[df['days_sampled'] > 0]

print(df.columns)
N = df.shape[0]
sensor_id = df['Sensor_ID'].values.reshape(N,1)

spores = df['spores'].values.reshape(N,1)

Collection_end_date = [dt.datetime.strptime(d, '%m/%d/%y') for d in df['Collection_end_date'].values]
Collection_start_date = [dt.datetime.strptime(d, '%m/%d/%y') for d in df['Collection_start_date'].values] 
end_day = date_to_day(Collection_end_date).reshape(N,1)
beg_day = date_to_day(Collection_start_date).reshape(N,1)
days_sampled = end_day - beg_day

spores /= days_sampled
if use_log:
    spores = np.log(spores+1)

center_date = end_day - 0.5 * days_sampled
days_offset = np.abs(center_date - center_date.T)

max_day_offset = np.maximum(np.maximum(np.abs(end_day - end_day.T),np.abs(end_day - beg_day.T)),
    np.maximum(np.abs(beg_day - end_day.T), np.abs(beg_day - beg_day.T))).astype(np.int32).flatten()
min_day_offset = np.minimum(np.minimum(np.abs(end_day - end_day.T),np.abs(end_day - beg_day.T)),
    np.minimum(np.abs(beg_day - end_day.T), np.abs(beg_day - beg_day.T))).astype(np.int32).flatten()

# do an intersect over union (IoU) for time overlap
time_overlap = (np.maximum(0, np.minimum(end_day, end_day.T) - np.maximum(beg_day, beg_day.T)) / \
    (np.maximum(end_day, end_day.T) - np.minimum(beg_day, beg_day.T)))

Lat = df['Latitude'].values.reshape(N,1)
Long = df['Longitude'].values.reshape(N,1)
DistMat = calculate_spherical_distance(Lat, Long, Lat.T, Long.T).flatten()

xy = (spores * spores.T).flatten()
x2 = (np.square(spores) * np.ones((1,N))).flatten()
y2 = (np.ones((N,1)) * np.square(spores.T)).flatten()

# Get overview over data first...
day_bin_sz = 5
un_day_j, inds = np.unique((np.round(center_date/day_bin_sz) * day_bin_sz).astype(np.int32), return_inverse=True)
un_sens_id, inds3 = np.unique(sensor_id, return_inverse=True)
inds4 = inds3 * un_day_j.size + inds
im = np.zeros((un_day_j.size * un_sens_id.size, 3), np.uint8)

numOccs = np.bincount(inds4,minlength=im.shape[0])
numHits = np.bincount(inds4,minlength=im.shape[0],weights=spores.flatten())
im[(numOccs>0)*(numHits > 0),0] = 255 # red
im[(numOccs>0)*(numHits == 0),1] = 255 # green
im = im.reshape((un_sens_id.size,un_day_j.size, 3))

fig, axes = plt.subplots()
plt.imshow(im,)
plt.show()

if 0:
    fig, axes = plt.subplots()
    ct_ave = np.zeros(un_day_j.size)
    for i in range(un_day_j.size):
        inds2 = np.where(inds == i)[0]
        ct_ave[i] = np.sum(spores[inds2]>0) 

    plt.plot(un_day_j, ct_ave)
    plt.show()

# Do the spaial ACF
spatial_pairs = np.where(((sensor_id != sensor_id.T) * (time_overlap>0.5) * 
    (center_date > 0*22100) * (center_date < 22175)).flatten())[0]
spatial_bin_sz_meters = 5
s_offset = (np.round(DistMat[spatial_pairs] / spatial_bin_sz_meters) * spatial_bin_sz_meters).astype(np.int32)
un_dists, inds = np.unique(s_offset, return_inverse=True)

ACF_dist = np.zeros(un_dists.size)
Ndatapts = np.zeros(un_dists.size)
for i in range(un_dists.size):
    inds2 = np.where(inds == i)[0]
    ACF_dist[i] = np.sum(xy[spatial_pairs[inds2]]) / np.sqrt(np.sum(x2[spatial_pairs[inds2]])*np.sum(y2[spatial_pairs[inds2]]))
    Ndatapts[i] = inds2.size

fig, ax = plt.subplots()
plt.plot(un_dists, ACF_dist)
if 1:
    ax2 = ax.twinx()
    plt.plot(un_dists, Ndatapts,'r')
ax.set_xlim(left=0,right=300)# if use_log else 50)
#ax.set_ylim(bottom=0)
#ax.set_xlabel('#Days apart for measurements on the same sensor')
#ax.set_ylabel(f'Auto correlation {"log([spore count] + 1)" if use_log else "spore count"}')
plt.show()


# do the temporal ACF 
same_sensor = np.where(((sensor_id == sensor_id.T) * (days_offset>0)).flatten())[0]
days_offset = days_offset.flatten()

if 1:
    M = max_day_offset.max() + 1
    xy_sum = np.zeros(M)
    Ndatapts = np.zeros(M)
    x2_sum = 1E-10*np.ones(M) 
    y2_sum = 1E-10*np.ones(M) 
    for i in same_sensor:
        inds = np.arange(min_day_offset[i],max_day_offset[i]+1)
        xy_sum[inds] += xy[i]
        x2_sum[inds] += x2[i]
        y2_sum[inds] += y2[i]
        Ndatapts[inds] += 1

    ACF_time = xy_sum / np.sqrt(x2_sum * y2_sum)
    un_days = np.arange(M)
else:
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
if 0:
    ax2 = ax.twinx()
    plt.plot(un_days, Ndatapts,'r')
ax.set_xlim(left=0,right=150 if use_log else 50)
ax.set_ylim(bottom=0)
ax.set_xlabel('#Days apart for measurements on the same sensor')
ax.set_ylabel(f'Auto correlation {"log([spore count] + 1)" if use_log else "spore count"}')
plt.show()
d = 1






