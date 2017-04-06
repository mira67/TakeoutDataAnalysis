#Detection customer dining locations
#Author: Qi Liu
#Email: qliu.hit@gmail.com

import ctypes
import os
import sys
import time

if getattr(sys, 'frozen', False):
  # Override dll search path.
  ctypes.windll.kernel32.SetDllDirectoryW('C:/Users/ngj/AppData/Local/Continuum/Anaconda3/Library/bin/')
  # Init code to load external dll
  ctypes.CDLL('mkl_avx2.dll')
  ctypes.CDLL('mkl_def.dll')
  ctypes.CDLL('mkl_vml_avx2.dll')
  ctypes.CDLL('mkl_vml_def.dll')

  # Restore dll search path.
  ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)

import psycopg2
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

#import method
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import multiprocessing as mp

path = 'E:/myprojects/takeout/code/'

#connect to database and get cursor
try:
    conn = psycopg2.connect("dbname='urbandata' user='postgres' host='localhost' password='1234'")

except:
    print "I am unable to connect to the database"
    
cur = conn.cursor()

sql = """
SELECT rates.shop_id, count(*) as sfreq, avg(to_number(rates.cost_time,'999')), shops.wgs_lat, shops.wgs_lon
FROM postgres.baidu_takeout_rating as rates
LEFT JOIN baidu_takeout_shops as shops ON shops.shop_id = rates.shop_id 
WHERE rates.pass_uid = %(user_id)s
GROUP BY rates.shop_id, shops.wgs_lat, shops.wgs_lon
ORDER BY sfreq;
"""

kms_per_radian = 6371.0088
epsilon = 3.5 / kms_per_radian

def patternDetection(user):
    #get user data, str(user)
    try:
        cur.execute(sql, {'user_id': str(user)})
        rows = cur.fetchall() 
    except:
        print "I am not able to query!"
        
    res = np.array(rows, dtype=np.float)
    #detect locations
    X = res[:,3:5]
    print type(X)
    
    db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(X))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters)
    #plot
    plotResult(user,labels, X, core_samples_mask, n_clusters)
                
    print 'ok'
    
def plotResult(user,labels, X, core_samples_mask, n_clusters):
    # Black removed and is used for noise instead.
    fig, ax = plt.subplots(figsize=(10,20))
    m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
            lat_0=39.905960083, lon_0=116.391242981,
            llcrnrlon=116.185913, llcrnrlat= 39.754713, urcrnrlon=116.552582, urcrnrlat=40.027614)
    
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
    m.drawcoastlines()
    m.readshapefile(path+'roads', 'bjroads')
    
    x, y = m(X[:,1], X[:,0])
    size = 5
    #m.plot(x, y, 'o', markersize=size, color='#f45642', alpha=0.8)
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        x, y = m(xy[:,1], xy[:,0])
        m.plot(x,y, 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=10)
    
        xy = X[class_member_mask & ~core_samples_mask]
        x, y = m(xy[:,1], xy[:,0])
        m.plot(x,y, '^', markerfacecolor=col,
                markeredgecolor='k', markersize=8)
    
    plt.title('Estimated number of clusters: %d' % n_clusters)
    #plt.show()
    
    plt.savefig(path+'/locations_detection/'+user+'_dbscan.png',bbox_inches='tight')
    plt.close()
        
    
if __name__ == '__main__':
        
    print 'Running Local Stage...'
    users = pd.read_excel(path+'topusers.xlsx'); 
    userList = users['user'].tolist()
    userList = map(str, userList)#seems only string list works for pool map

    #profiling 1
    start = time.time()
    #patternDetection('3337871')
    pool = mp.Pool(1)
    results = pool.map(patternDetection, userList)
    
    end = time.time()
    runtime = end - start
    msg = "Pattern Detection Multi-Processing Took {time} seconds to complete"
    print(msg.format(time=runtime))    