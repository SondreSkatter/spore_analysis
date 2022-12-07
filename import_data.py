import pandas as pd
import numpy as np
 

def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371000):
    # Convert degrees to radians
    coordinates = lat1, lon1, lat2, lon2
    def radians(c):
        return c*np.pi/180
    phi1, lambda1, phi2, lambda2 = [
        radians(c) for c in coordinates
    ]  
    
    # Apply the haversine formula
    a = (np.square(np.sin((phi2-phi1)/2)) + np.cos(phi1) * np.cos(phi2) * 
         np.square(np.sin((lambda2-lambda1)/2)))
    d = 2*r*np.arcsin(np.sqrt(a))
    return d


# Data.tsv is stored locally in the
# same directory as of this python file
df = pd.read_csv('data/Roots_data_2021_2022.txt',sep = '\t')

print(df.columns)
spores = df['spores']
sensor_id = df['Sensor_ID']
Collection_end_date = df['Collection_end_date']
days_sampled = df['days_sampled']

un_id, ind = np.unique(sensor_id, return_inverse=True)
N = ind.size

Lat = df['Latitude'].values.reshape(N,1)
Long = df['Longitude'].values.reshape(N,1)
DistMat = calculate_spherical_distance(Lat, Long, Lat.T, Long.T)


distanceMat = np.zeros((N,N))
df2 = pd.read_csv('data/distance-matrix_2022.txt',sep = '\t')
row_ids = df2['V1']
sortInd = -np.ones(N,np.uint32)
for i in range(N):
    try:
        sortInd[i] = np.where(un_id[i] == row_ids)[0][0]
    except:
        print(f'No match found for {un_id[i]}')


print(df)



