CREATE TABLE baidu_usertest AS (
SELECT rates.shop_id, count(*) as sfreq, avg(to_number(rates.cost_time,'999')), shops.wgs_lat, shops.wgs_lon
FROM postgres.baidu_takeout_rating as rates
LEFT JOIN baidu_takeout_shops as shops ON shops.shop_id = rates.shop_id 
WHERE rates.pass_uid = '673426103'
GROUP BY rates.shop_id, shops.wgs_lat, shops.wgs_lon
ORDER BY sfreq
)