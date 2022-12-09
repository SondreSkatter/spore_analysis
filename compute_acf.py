import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime as dt
from matplotlib import pyplot as plt

use_log = True
show_image = False
do_spatial = True
do_temporal = True
begDate = dt.datetime(2022,4,15)
endDate = dt.datetime(2022,7,15)
begDate = dt.datetime(2021,4,20)
endDate = dt.datetime(2023,6,10)

def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371000):
    # Convert degrees to radians
    phi1, lambda1, phi2, lambda2 = [c*np.pi/180 for c in (lat1, lon1, lat2, lon2)]      
    # Apply the haversine formula
    a = (np.square(np.sin((phi2-phi1)/2)) + np.cos(phi1) * np.cos(phi2) * 
         np.square(np.sin((lambda2-lambda1)/2)))
    d = 2*r*np.arcsin(np.sqrt(a))
    return d

def date_to_day(dates):
    if type(dates) == list:
        return np.array([d.timestamp()/24/3600 for d in dates])
    else:
        return dates.timestamp()/24/3600

def day_to_date(dates):
    if type(dates) == list:
        return np.array([dt.datetime.fromtimestamp(d*24*3600) for d in dates])
    else:
        return dt.datetime.fromtimestamp(dates*24*3600)

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

#spores = (spores > 0).astype(np.float)

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


Mean = np.mean(spores)
xy = ((spores - Mean) * (spores - Mean).T).flatten()
x2 = (np.square(spores-Mean) * np.ones((1,N))).flatten()
y2 = (np.ones((N,1)) * np.square(spores.T - Mean)).flatten()
x = (spores * np.ones((1,N))).flatten()
y = (np.ones((N,1)) * spores.T).flatten()

if show_image:
    # Get overview over data first...
    day_bin_sz = 5
    un_day_j, inds = np.unique((np.round(center_date/day_bin_sz) * day_bin_sz).astype(np.int32), return_inverse=True)
    un_sens_id, inds3 = np.unique(sensor_id, return_inverse=True)
    inds4 = inds3 * un_day_j.size + inds
    im = 127 * np.ones((un_day_j.size * un_sens_id.size, 3), np.uint8)

    numOccs = np.bincount(inds4,minlength=im.shape[0])
    numHits = np.bincount(inds4,minlength=im.shape[0],weights=spores.flatten())
    im[(numOccs>0)*(numHits > 0),:] = np.array(((255, 0, 0)),np.uint8) # red
    im[(numOccs>0)*(numHits == 0),:] = np.array(((0, 255, 0)),np.uint8) # green
    im = im.reshape((un_sens_id.size,un_day_j.size, 3))

    fig, axes = plt.subplots()
    ticks = np.arange(start=0,stop=un_day_j.size,step=4)
    plt.imshow(im,aspect='auto')
    axes.set_xticks(ticks)
    axes.set_xticklabels([dt.datetime.fromtimestamp(d*3600*24).strftime('%b %d') for d in un_day_j[ticks]],rotation = 45)
    ticks = np.arange(un_sens_id.size,step=2)
    axes.set_yticks(ticks)
    axes.set_yticklabels(un_sens_id[ticks], fontsize=7)
    axes.set_title('Detection: red, Clear: green, No test data: gray')
    plt.show()

if 0:
    fig, axes = plt.subplots()
    ct_ave = np.zeros(un_day_j.size)
    for i in range(un_day_j.size):
        inds2 = np.where(inds == i)[0]
        ct_ave[i] = np.sum(spores[inds2]>0) 

    plt.plot(un_day_j, ct_ave)
    plt.show()

if do_spatial:
    # Do the spaial ACF
    spatial_pairs = np.where(((sensor_id != sensor_id.T) * (time_overlap > 0.) * 
        (center_date > date_to_day(begDate)) * (center_date < date_to_day(endDate))).flatten())[0]
    doEqualBinSizes = True
    useKernel = True
    if useKernel:
        # Using a Gaussian kernel for each distance
        
        sigma_meters = 5
        resolution_meters = 10
        max_dist = 1000

        spatial_pairs = spatial_pairs[DistMat[spatial_pairs] < max_dist]
        Bins = np.arange(resolution_meters/2,max_dist,resolution_meters)
        M = Bins.size
        xAll = np.zeros(M)
        yAll = np.zeros(M)
        x2All = np.zeros(M)
        y2All = np.zeros(M)
        xyAll = np.zeros(M)
        nAll = np.zeros(M)
        for i in spatial_pairs:
            Profile = stats.norm.pdf(Bins, DistMat[i] , sigma_meters)
            Profile = Profile / np.sum(Profile)
            xAll += Profile * x[i]
            yAll += Profile * y[i]
            nAll += Profile

        Mean = 0.5 * (xAll + yAll) / nAll

        for i in spatial_pairs:
            Profile = stats.norm.pdf(Bins, DistMat[i] , sigma_meters)
            Profile = Profile / np.sum(Profile)            
            x2All += Profile * np.square(x[i] - Mean)
            y2All += Profile * np.square(y[i] - Mean)
            xyAll += Profile * (x[i] - Mean) * (y[i] - Mean)            

        x2All /= nAll
        y2All /= nAll
        xyAll /= nAll
        ACF_dist = xyAll / np.sqrt(x2All*y2All)
        un_dists = Bins
        Ndatapts = nAll


    elif doEqualBinSizes:
        # do equal number of points in each bin
        sortInd = np.argsort(DistMat[spatial_pairs])
        M = 200 # num points per bin
        numBins = int(sortInd.size / M)
        ACF_dist = np.zeros(numBins)
        un_dists = np.zeros(numBins)
        Ndatapts = np.zeros(numBins)        
        for i in range(numBins):
            inds2 = spatial_pairs[sortInd[np.arange(i*M, (i+1)*M)]]
            xAll = np.append(x[inds2], y[inds2])
            Mean = xAll.mean()

            #ACF_dist[i] = np.sum(xy[inds2]) / np.sqrt(np.sum(x2[inds2])*np.sum(y2[inds2]))
            ACF_dist[i] = np.mean((x[inds2]-Mean)*(y[inds2]-Mean)) / np.mean(np.square(xAll - Mean))
            un_dists[i] = np.mean(DistMat[inds2])
            Ndatapts[i] = inds2.size
    else:
        spatial_bin_sz_meters = 25
        s_offset = (np.round(DistMat[spatial_pairs] / spatial_bin_sz_meters) * spatial_bin_sz_meters).astype(np.int32)
        un_dists, inds = np.unique(s_offset, return_inverse=True)

        ACF_dist = np.zeros(un_dists.size)
        Ndatapts = np.zeros(un_dists.size)
        for i in range(un_dists.size):
            inds2 = spatial_pairs[np.where(inds == i)[0]]
            xAll = np.append(x[inds2], y[inds2])
            Mean = xAll.mean()

            #ACF_dist[i] = np.sum(xy[inds2]) / np.sqrt(np.sum(x2[inds2])*np.sum(y2[inds2]))
            ACF_dist[i] = np.mean((x[inds2]-Mean)*(y[inds2]-Mean)) / np.mean(np.square(xAll - Mean))
            Ndatapts[i] = inds2.size

    fig, ax = plt.subplots()
    plt.plot(un_dists, ACF_dist)
    if useKernel or not doEqualBinSizes:
        ax2 = ax.twinx()
        plt.plot(un_dists, Ndatapts,'r')
    ax.set_xlim(left=0,right=1000)# if use_log else 50)
    ax.set_ylim(bottom=0)
    #ax.set_xlabel('#Days apart for measurements on the same sensor')
    #ax.set_ylabel(f'Auto correlation {"log([spore count] + 1)" if use_log else "spore count"}')
    plt.show()

if do_temporal:
    # do the temporal ACF 
    same_sensor = np.where(((sensor_id == sensor_id.T) * (days_offset>0)).flatten())[0]
    
    if 1:
        M = max_day_offset.max() + 1
        Ndatapts = np.zeros(M)
        if 1:
            ACF_time = np.zeros(M)
            for i in range(M):
                inds = same_sensor[(i>=min_day_offset[same_sensor])*(i<=max_day_offset[same_sensor])]
                xAll = np.append(x[inds], y[inds])
                Mean = xAll.mean()
                ACF_time[i] = np.mean((x[inds]-Mean)*(y[inds]-Mean)) / np.mean(np.square(xAll - Mean))
                Ndatapts[i] = inds.size
        else:
            xy_sum = np.zeros(M)
            x2_sum = 1E-10*np.ones(M) 
            y2_sum = 1E-10*np.ones(M) 
            for i in same_sensor:
                inds = np.arange(min_day_offset[i],max_day_offset[i]+1)
                #xMean = 0.5*np.mean()
                xy_sum[inds] += xy[i]
                x2_sum[inds] += x2[i]
                y2_sum[inds] += y2[i]
                Ndatapts[inds] += 1

            ACF_time = xy_sum / np.sqrt(x2_sum * y2_sum)
        un_days = np.arange(M)
    else:
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
    if 0:
        ax2 = ax.twinx()
        plt.plot(un_days, Ndatapts,'r')
    ax.set_xlim(left=0,right=150 if use_log else 50)
    #ax.set_ylim(bottom=0)
    ax.set_xlabel('#Days apart for measurements on the same sensor')
    ax.set_ylabel(f'Auto correlation {"log([spore count] + 1)" if use_log else "spore count"}')
    plt.show()
d = 1






