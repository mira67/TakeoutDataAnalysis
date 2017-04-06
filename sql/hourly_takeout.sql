CREATE TABLE baidu_user3337871_weekend AS
SELECT
  EXTRACT(HOUR FROM to_timestamp(rt.arrive_time,'YYYY-MM-DD HH24:MI')) AS hour,
  COUNT(*) AS ratec,
  avg(sp.wgs_lat) as clat,
  avg(sp.wgs_lon) as clon
FROM baidu_takeout_rating as rt
LEFT JOIN baidu_daily_rates as dy
ON date(to_timestamp(rt.arrive_time,'YYYY-MM-DD HH24:MI')) = dy.day
LEFT JOIN baidu_takeout_shops as sp
ON rt.shop_id = sp.release_id
WHERE rt.pass_uid = '3337871' AND dy.weekday = 'false'
GROUP BY hour,dy.weekday
ORDER BY hour
