#Detection of customer dining locations
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
from sklearn.cluster import KMeans
import multiprocessing as mp

#customized
from mean_shift_n import MeanShift

path = 'E:/myprojects/takeout/code/python/'
resPath = 'E:/myprojects/takeout/results/'
Outpath = 'E:/myprojects/takeout/results/meanshift_test_0803/'
patternFile = 'pattern_features_ex_0831.csv'
shopListFile = 'shop_list_ex_0831.csv'

#hybrid parameter
nk = 2  
epsPercentile = 99  

#connect to database and get cursor
try:
    conn = psycopg2.connect("dbname='urbandata' user='postgres' host='localhost' password='1234'")

except:
    print("I am unable to connect to the database")
    
cur = conn.cursor()

sql = """
SELECT rates.shop_id, count(*) as sfreq, avg(to_number(rates.cost_time,'999')), shops.wgs_lat, shops.wgs_lon
FROM postgres.baidu_takeout_rating_ex as rates
LEFT JOIN baidu_takeout_shops_extend as shops ON shops.shop_id = rates.shop_id 
WHERE rates.pass_uid = %(user_id)s
GROUP BY rates.shop_id, shops.wgs_lat, shops.wgs_lon
ORDER BY rates.shop_id;
"""

sql_user_group = """
SELECT rates.shop_id, rates.sfrom, rates.arrive_time, rates.cost_time, rates.score, rates.service_score, rates.dish_score, 
shops.takeout_price, shops.takeout_cost, shops.average_score,shops.wgs_lat, shops.wgs_lon
FROM postgres.baidu_takeout_rating as rates
LEFT JOIN baidu_takeout_shops as shops ON shops.shop_id = rates.shop_id 
WHERE rates.pass_uid = %(user_id)s
AND
date(rates.arrive_time) between '2016-01-01' and '2017-02-28'
ORDER BY rates.pass_uid;
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
    baidu_takeout_rating_ex
WHERE pass_uid = %(user_id)s
GROUP BY 
    shop_id, pass_uid
ORDER BY shop_id
"""

"""Configurations"""
kms_per_radian = 6371.0088
epsilon = 1.0 / kms_per_radian

feature_columns = ['weekday','saturday','sunday','h00','h01','h02','h03','h04','h05','h06','h07','h08','h09','h10','h11',
'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','user_id']

#selecting method type: basic dbscan or hybrid method (dbscan + kmeans)
dbscan = 0
hybrid = 0#hybrid with auto-determined eps 
meanshift = 1
def patternDetection(user):
    #get user data, str(user)
    user = user[:-2]#temporal fix for the format of user id with '.0'
    try:    
        cur.execute(sql, {'user_id': str(user)})
        rows = cur.fetchall() 
        res = np.array(rows)
        shopList = res[:,0]
    except:
        print("I am not able to query!")
        
    if dbscan == 1:
        #detect locations
        X = np.float64(res[:,3:5])
        epsilon = 3.0 / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(X))
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
        #"""plot and save figure"""
        #plotResult(user,labels, X, core_samples_mask, n_clusters, 0)
        
        """extract pattern features"""
        patternFeature(user, shopList, labels, core_samples_mask, n_clusters)
        
        
    if meanshift == 1:
        try:
            #detect locations
            X = np.float64(res[:,3:5])
            X_radians = np.radians(X)
            
            X_dt = np.float64(res[:,2])
            dt_size = X_dt.size
            X_dt = X_dt.reshape((dt_size,1))
            
            #bandwidth = estimate_bandwidth(X_radians, quantile=0.2)
            bandwidth = 0.00047088304#=3km,0.00031392202
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all = False, kernel='takeout', gamma=1.0, computed_weights = True, weights=X_dt,n_jobs =1)
            ms.fit(X_radians)
    
            labels = ms.labels_
            
            large_labels = labels[np.where(labels>-1)]
            
            unique, counts = np.unique(large_labels, return_counts=True)
            
            ind, = np.where(counts==1)
            
            cluster_centers = np.degrees(ms.cluster_centers_)
            
            if ind.size > 0:
                cluster_centers = np.delete(cluster_centers,ind,axis=0)
                for lid in ind:
                    labels[labels == unique[lid]] = -1
            
            #print(cluster_centers)
            
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
            #"""plot and save figure"""
            #plotResultBasic(user,labels, X, n_clusters,cluster_centers)
            
            """extract pattern features"""
            patternFeatureUpdate(user, shopList, labels, n_clusters,cluster_centers)
        except:
            print("Not able to run for user: ", user)
        
    if hybrid ==1:
    
        try:
            #detect locations
            X = np.float64(res[:,3:5])#spatial attribute
            feaX = np.radians(X)
            
            #auto-determine eps
            '''
            nbrs = NearestNeighbors(n_neighbors=nk, algorithm='auto',metric='haversine').fit(feaX)
            distances, indices = nbrs.kneighbors(feaX)
            realDistance =  distances*kms_per_radian
            Eps = np.percentile(realDistance[:,nk-1],epsPercentile)
            '''
            
            epsilon = 3.0 / kms_per_radian
    
            db = DBSCAN(eps=epsilon, min_samples=nk, algorithm='ball_tree', metric='haversine').fit(feaX)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            #kmeans to locate centroid and compute center to center distance as profile attributes
            hubCenters = np.zeros((n_clusters, 4))#lat, lon, center avg delivery time, non-center delivery time
            
            #for each hub, using kmeans to determine the centroid
            for nc in range(0,n_clusters):
                class_member_mask = (labels == nc)
                hubData = res[class_member_mask]
                hubFeature = np.float64(hubData[:,1:3])#review count & delivery time
                hubLocation = np.float64(hubData[:,3:5])
                #apply K-Means, assume only noise and dense shops, K = 2
                K = 2
                km = KMeans(n_clusters=K, random_state=0).fit(hubFeature)
                kmLabels = km.labels_
                #pick the cluster center with less delivery time as hub center
                kmCenters = km.cluster_centers_
                hubCenterIdx = np.argmin(kmCenters[:,1])
                hubCenter = kmCenters[hubCenterIdx,:]#delivery and review count center
                
                #get non-center
                nonCenterIdx = np.argmax(kmCenters[:,1])
                nonCenter = kmCenters[nonCenterIdx,:]#delivery and review count center
                
                #extract hub center location centroid, calculated weighted centroid
                hubCenterLocs = hubLocation[kmLabels == hubCenterIdx]
                hubDeliveryTime = hubFeature[kmLabels == hubCenterIdx,1]
                #handling zeros bad value
                tempm = np.mean(hubDeliveryTime[hubDeliveryTime > 0])
                # Assign the median to the zero elements 
                hubDeliveryTime[hubDeliveryTime == 0] = tempm
                
                tempx = np.divide(hubCenterLocs[:,0],hubDeliveryTime)
                tempy = np.divide(hubCenterLocs[:,1],hubDeliveryTime)
                temp = np.sum(np.reciprocal(hubDeliveryTime))
                x = np.sum(tempx)/temp
                y = np.sum(tempy)/temp
                hubCenters[nc,0:2] = np.array([x,y])#assign lat, lon
                hubCenters[nc,2] = hubCenter[1]#assign delivery time
                hubCenters[nc,3] = nonCenter[1]#assign non-center delivery time        
            #"""plot and save figure"""
            #plotResult(user,labels, X, core_samples_mask, n_clusters, hubCenters)
            patternFeature(user, shopList, labels, core_samples_mask, n_clusters, hubCenters)
        except:
            print("Not able to run for user: ", user)

#extract temporal features for each pattern
def patternFeatureUpdate(user, shopList, labels, n_clusters, hubCenters):
    #obtain feature list for one pattern, attach user_id information + features + shop_id
    try:
        cur.execute(sql_feature, {'user_id': str(user)})          
        rows = cur.fetchall() 
    except:
        print("I am not able to query!")
        
    features = np.array(rows, dtype=np.float)
    #aggregate for each cluster
    for nc in range(0,n_clusters):
        #pattern_id increase
        class_member_mask = (labels == nc)
        pattern = features[class_member_mask]
        #record shop list for each pattern
        shop_list = shopList[class_member_mask]
        shop_list = np.insert(shop_list,0,user)
        shop_list = np.insert(shop_list,0,user+'0'+str(nc))
        nshop = len(shop_list)
        with open(resPath+shopListFile,'ab') as f_handle:
            np.savetxt(f_handle, shop_list.reshape((1,nshop)), delimiter=',',fmt='%s')
        
        pattern_feature = np.sum(pattern[:,1:28], axis=0)
        #scale columns to ratio
        total = np.sum(pattern_feature[0:3])
        pattern_feature = pattern_feature/total
        pattern_feature = np.append(pattern_feature,int(user))
        pattern_feature = np.append(pattern_feature,int(user)*100+nc)
        pattern_feature = np.append(pattern_feature,hubCenters[nc,:])
        nfea = len(pattern_feature)
        #add user_id to end and record to file
        with open(resPath+patternFile,'ab') as f_handle:
            np.savetxt(f_handle, pattern_feature.reshape((1,nfea)), delimiter=',',fmt='%s')

    print('Come on!')
        
def patternFeature(user, shopList, labels, n_clusters, hubCenters):
    #obtain feature list for one pattern, attach user_id information + features + shop_id
    try:
        cur.execute(sql_feature, {'user_id': str(user)})          
        rows = cur.fetchall() 
    except:
        print("I am not able to query!")
        
    features = np.array(rows, dtype=np.float)
    #aggregate for each cluster
    for nc in range(0,n_clusters):
        #pattern_id increase
        class_member_mask = (labels == nc)
        pattern = features[class_member_mask]
        #record shop list for each pattern
        shop_list = shopList[class_member_mask]
        shop_list = np.insert(shop_list,0,user)
        shop_list = np.insert(shop_list,0,user+'0'+str(nc))
        nshop = len(shop_list)
        with open(resPath+shopListFile,'a') as f_handle:
            np.savetxt(f_handle, shop_list.reshape((1,nshop)), delimiter=',',fmt='%s')
        
        pattern_feature = np.sum(pattern[:,1:28], axis=0)
        #scale columns to ratio
        total = np.sum(pattern_feature[0:3])
        pattern_feature = pattern_feature/total
        pattern_feature = np.append(pattern_feature,int(user))
        pattern_feature = np.append(pattern_feature,int(user)*100+nc)
        pattern_feature = np.append(pattern_feature,hubCenters[nc,:])
        nfea = len(pattern_feature)
        #add user_id to end and record to file
        with open(resPath+patternFile,'a') as f_handle:
            np.savetxt(f_handle, pattern_feature.reshape((1,nfea)), delimiter=',')
            #np.savetxt(f_handle, pattern, delimiter=',')
    #print 'Come on!'
    
def patternClustering(pattern):
    #read in features from database for all patterns and cluster
    #BIC to determine the proper number of clusters
    
    print('Just Do It!')
    
def plotResult(user,labels, X, core_samples_mask, n_clusters, hubCenters):
    # Black removed and is used for noise instead.
    fig, ax = plt.subplots(figsize=(4,8))
    m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
            lat_0=39.905960083, lon_0=116.391242981,
            llcrnrlon=116.185913, llcrnrlat= 39.754713, urcrnrlon=116.552582, urcrnrlat=40.027614)
    msize = 5
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
                markeredgecolor='k', markersize=msize)
    
        xy = X[class_member_mask & ~core_samples_mask]
        x, y = m(xy[:,1], xy[:,0])
        m.plot(x,y, '^', markerfacecolor=col,
                markeredgecolor='k', markersize=msize)
                
        #draw hubCenters
        if k != -1:
            x,y = m(hubCenters[k,1],hubCenters[k,0])#lat,lon
            m.plot(x,y, 'X', markerfacecolor='r',
                markeredgecolor='r', markersize=msize)
    
    plt.title('Estimated number of clusters: %d' % n_clusters)
    #plt.show()
    
    plt.savefig(Outpath+user+'_dbscan.png',bbox_inches='tight',dpi=200)
    plt.close()
    
def plotResultBasic(user,labels, X, n_clusters,cluster_centers):
    # Black removed and is used for noise instead.
    fig, ax = plt.subplots(figsize=(4,8))
    m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
            lat_0=39.905960083, lon_0=116.391242981,
            llcrnrlon=116.185913, llcrnrlat= 39.754713, urcrnrlon=116.552582, urcrnrlat=40.027614)
    msize = 5
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
    
        xy = X[class_member_mask]
        x, y = m(xy[:,1], xy[:,0])
        m.plot(x,y, 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=msize)
                
        x,y = m(cluster_centers[k,1],cluster_centers[k,0])#lat,lon
        m.plot(x,y, 'X', markerfacecolor='r',
                markeredgecolor='r', markersize=msize)
    
    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()
    
    plt.savefig(Outpath+user+'_meanshift_basic.png',bbox_inches='tight',dpi=200)
    #plt.close()
    
def main():
    print('Running pattern...')
    #users = pd.read_excel(path+'hub_detection_test_0703.xlsx'); 
    users = pd.read_csv(path+'baidu_user45_extend.csv')
    userList = users['pass_uid'].tolist()

    userList = map(str, userList)#seems only string list works for pool map
    
    #profiling 1
    start = time.time()
    #patternDetection('1191278995')
    with mp.Pool(3) as pool:
        results = pool.map(patternDetection, userList)
    end = time.time()
    runtime = end - start
    msg = "Pattern Detection Multi-Processing Took {time} seconds to complete"
    print(msg.format(time=runtime)) 
    
if __name__ == "__main__":
    # execute only if run as a script
    main()