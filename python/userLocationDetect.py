#Detection customer dining locations
#Author: Qi Liu
#Email: qliu.hit@gmail.com

import ctypes
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
from mpl_toolkits.basemap import Basemap

#import method
import numpy as np
from sklearn.cluster import DBSCAN
import multiprocessing as mp

import csv

path = 'E:/myprojects/takeout/code/python/'
Outpath = 'E:/myprojects/takeout/results/locations_detection_auto_eps_nn2_0430/'

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

sql_feature = """
SELECT 
    shop_id,
    sum(case when dayOfweek between 1 and 5 then 1 else 0 end) as weekday,
    sum(case when dayOfweek = 6 then 1 else 0 end) as saturday,
    sum(case when dayOfweek = 0 then 1 else 0 end) as sunday,
    sum(case when hourOfday = 0 then 1 else 0 end) as h00,
    sum(case when hourOfday = 1 then 1 else 0 end) as h01,
    sum(case when hourOfday = 2 then 1 else 0 end) as h02,
    sum(case when hourOfday = 3 then 1 else 0 end) as h03,
    sum(case when hourOfday = 4 then 1 else 0 end) as h04,
    sum(case when hourOfday = 5 then 1 else 0 end) as h05,
    sum(case when hourOfday = 6 then 1 else 0 end) as h06,
    sum(case when hourOfday = 7 then 1 else 0 end) as h07,
    sum(case when hourOfday = 8 then 1 else 0 end) as h08,
    sum(case when hourOfday = 9 then 1 else 0 end) as h09,
    sum(case when hourOfday = 10 then 1 else 0 end) as h10,
    sum(case when hourOfday = 11 then 1 else 0 end) as h11,
    sum(case when hourOfday = 12 then 1 else 0 end) as h12,
    sum(case when hourOfday = 13 then 1 else 0 end) as h13,
    sum(case when hourOfday = 14 then 1 else 0 end) as h14,
    sum(case when hourOfday = 15 then 1 else 0 end) as h15,
    sum(case when hourOfday = 16 then 1 else 0 end) as h16,
    sum(case when hourOfday = 17 then 1 else 0 end) as h17,
    sum(case when hourOfday = 18 then 1 else 0 end) as h18,
    sum(case when hourOfday = 19 then 1 else 0 end) as h19,
    sum(case when hourOfday = 20 then 1 else 0 end) as h20,
    sum(case when hourOfday = 21 then 1 else 0 end) as h21,
    sum(case when hourOfday = 22 then 1 else 0 end) as h22,
    sum(case when hourOfday = 23 then 1 else 0 end) as h23,
    pass_uid
FROM
    baidu_takeout_temporal
WHERE pass_uid = %(user_id)s
GROUP BY 
    shop_id, pass_uid
ORDER BY shop_id
"""

"""Configurations"""
kms_per_radian = 6371.0088
epsilon = 3.0 / kms_per_radian

feature_columns = ['weekday','saturday','sunday','h00','h01','h02','h03','h04','h05','h06','h07','h08','h09','h10','h11',
'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','user_id']

#selecting method type: basic dbscan or hybrid method (dbscan + kmeans)
dbscan = 0
hybrid = 1#hybrid with auto-determined eps 

def patternDetection(user):
    #get user data, str(user)
    try:
        cur.execute(sql, {'user_id': str(user)})
        rows = cur.fetchall() 
    except:
        print "I am not able to query!"
        
    res = np.array(rows)
    shopList = res[:,0]
    
    if dbscan == 1:
        #detect locations
        X = np.float64(res[:,3:5])#shop visited times and average delivery time
        
        db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(X))
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
        #"""plot and save figure"""
        plotResult(user,labels, X, core_samples_mask, n_clusters)
        
        """extract pattern features"""
        patternFeature(user, shopList, labels, core_samples_mask, n_clusters)
    
    if hybrid ==1:
        #detect locations
        X = np.float64(res[:,3:5])#shop visited times and average delivery time
        
        db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(X))
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
    print 'ok'

#extract temporal features for each pattern    
def patternFeature(user, shopList, labels, core_samples_mask, n_clusters):
    #obtain feature list for one pattern, attach user_id information + features + shop_id
    try:
        cur.execute(sql_feature, {'user_id': str(user)})
        rows = cur.fetchall() 
    except:
        print "I am not able to query!"
        
    features = np.array(rows, dtype=np.float)
    #aggregate for each cluster
    for nc in range(0,n_clusters):
        #pattern_id increase
        class_member_mask = (labels == nc)
        pattern = features[class_member_mask & core_samples_mask]
        #record shop list for each pattern
        shop_list = shopList[class_member_mask & core_samples_mask]
        shop_list = np.insert(shop_list,0,user)
        shop_list = np.insert(shop_list,0,user+'00'+str(nc))
        nshop = len(shop_list)
        with open(path+'pattern_shoplist.csv','a') as f_handle:
            np.savetxt(f_handle, shop_list.reshape((1,nshop)), delimiter=',',fmt='%s')
        
        pattern_feature = np.sum(pattern[:,1:28], axis=0)
        #scale columns to ratio
        total = np.sum(pattern_feature[0:3])
        pattern_feature = pattern_feature/total
        pattern_feature = np.append(pattern_feature,int(user))
        pattern_feature = np.append(pattern_feature,int(user)*100+nc)
        nfea = len(pattern_feature)
        #add user_id to end and record to file
        with open(path+'pattern_features.csv','a') as f_handle:
            np.savetxt(f_handle, pattern_feature.reshape((1,nfea)), delimiter=',')
    print 'Come on!'
    
def patternClustering(pattern):
    #read in features from database for all patterns and cluster
    #BIC to determine the proper number of clusters
    
    print 'Just Do It!'
    
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
    
    plt.savefig(Outpath+user+'_dbscan.png',bbox_inches='tight')
    plt.close()
        
    
if __name__ == '__main__':
        
    print 'Running pattern...'
    users = pd.read_excel(path+'new_dbscan_test_users_0430.xlsx'); 
    #users = pd.read_csv(path+'baidu_top200_user.csv');
    userList = users['pass_uid'].tolist()
    userList = map(str, userList)#seems only string list works for pool map

    #profiling 1
    start = time.time()
    #patternDetection('940389912')
    pool = mp.Pool(4)
    results = pool.map(patternDetection, userList)
    
    end = time.time()
    runtime = end - start
    msg = "Pattern Detection Multi-Processing Took {time} seconds to complete"
    print(msg.format(time=runtime))    