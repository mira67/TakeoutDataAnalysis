#exploratory analysis for user spatial patterns
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


#global timing 
start = time.time()

path = 'E:/myprojects/takeout/code/'

#1. connect to database
try:
    conn = psycopg2.connect("dbname='urbandata' user='postgres' host='localhost' password='1234'")

except:
    print "I am unable to connect to the database"
    
cur = conn.cursor()
     
#2. read in list of potential users from csv, generate from matlab
users = pd.read_excel(path+'topusers.xlsx');
#print users.head()

#3. got through each user, query visited shops (freq, avgTime, lat, lon)
sql = """
SELECT rates.shop_id, count(*) as sfreq, avg(to_number(rates.cost_time,'999')), shops.wgs_lat, shops.wgs_lon
FROM postgres.baidu_takeout_rating as rates
LEFT JOIN baidu_takeout_shops as shops ON shops.shop_id = rates.shop_id 
WHERE rates.pass_uid = %(user_id)s
GROUP BY rates.shop_id, shops.wgs_lat, shops.wgs_lon
ORDER BY sfreq;
"""

def plot_area(m,pos):
    x, y = m(pos[4], pos[3])
    size = 5
    m.plot(x, y, 'o', markersize=size, color='#f45642', alpha=0.8)
    plt.text(x,y,pos[1],fontsize=10,fontweight='medium',
                    ha='center',va='center',color='b')

#users = ['673426103'];

for index, row in users.iterrows():
    
    if index >= 289:
        print index
        user = str(row['user'])
    
        try:
            cur.execute(sql, {'user_id': user})
        except:
            print "I am not able to query!"
            
        rows = cur.fetchall() 
        
        #3.1 for each user , draw shops on map with freq and avgTime labels, on a beijing base map
        #westlimit=116.0431; southlimit=39.6586; eastlimit=116.7599; northlimit=40.1852
        fig, ax = plt.subplots(figsize=(10,20))
        m = Basemap(resolution='c', # c, l, i, h, f or None
                projection='merc',
                lat_0=39.905960083, lon_0=116.391242981,
                llcrnrlon=116.185913, llcrnrlat= 39.754713, urcrnrlon=116.552582, urcrnrlat=40.027614)
        
        m.drawmapboundary(fill_color='#46bcec')
        m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
        m.drawcoastlines()
        m.readshapefile(path+'roads', 'bjroads')
        
        for row in rows:
            plot_area(m,row)
        
        #plt.show()
    
        plt.savefig(path+'/figs/'+user+'.png',bbox_inches='tight')
        plt.close()

end = time.time()
runtime = end - start
msg = "Took {time} seconds to complete"
print(msg.format(time=runtime))
        
print 'TEST DONE'
#3.2 save figure for each user to a directory

#3.3 done and report total time






"""
    print "\nShow me the databases:\n"
    for row in rows:
        print "   ", row[1]
        """











