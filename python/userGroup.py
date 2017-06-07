#user group analysis
#Author: Qi Liu
#Date: 05/12/2017

import ctypes
import sys
import time
import multiprocessing as mp

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

#import method
import numpy as np
import csv

path = 'E:/myprojects/takeout/code/python/'
resPath = 'E:/myprojects/takeout/results/'
groupFile = 'userGroup11_single_0512.csv'

#connect to database and get cursor
try:
    conn = psycopg2.connect("dbname='urbandata' user='postgres' host='localhost' password='1234'")

except:
    print "I am unable to connect to the database"
    
cur = conn.cursor()

sql_user_group = """
SELECT rates.pass_uid,rates.shop_id, rates.sfrom, rates.arrive_time, rates.cost_time, rates.score, rates.service_score, rates.dish_score, 
shops.takeout_price, shops.takeout_cost, shops.average_score,shops.wgs_lat, shops.wgs_lon
FROM postgres.baidu_takeout_rating as rates
LEFT JOIN baidu_takeout_shops as shops ON shops.shop_id = rates.shop_id 
WHERE rates.pass_uid = %(user_id)s
AND
date(rates.arrive_time) between '2016-01-01' and '2017-02-28'
ORDER BY rates.pass_uid;
"""

def userGroupData(user):
    #get user data, str(user)
    try:    
        cur.execute(sql_user_group, {'user_id': str(user)})
        rows = cur.fetchall() 
        #res = np.array(rows)
        res = pd.DataFrame(rows)

    except:
        print "I am not able to query!"
    nfea = 13

    with open(resPath+groupFile,'a') as f_handle:
        res.to_csv(f_handle, header=False)
            
if __name__ == '__main__':
        
    print 'Running pattern...'
    users = pd.read_excel(path+'baidu_userGroup89.xlsx'); 
    #users = pd.read_csv(path+'baidu_userGroup89.csv');
    userList = users['pass_uid'].tolist()
    userList = map(str, userList)#seems only string list works for pool map

    #profiling 1
    start = time.time()
    userGroupData('2030904815')
    #pool = mp.Pool(3)
    #results = pool.map(userGroupData, userList)
    
    end = time.time()
    runtime = end - start
    msg = "User Group Data Extraction Multi-Processing Took {time} seconds to complete"
    print(msg.format(time=runtime))    