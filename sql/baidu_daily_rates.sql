CREATE TABLE baidu_daily_rates AS
SELECT
  date(to_timestamp(arrive_time,'YYYY-MM-DD HH24:MI')) AS day,
  EXTRACT(HOUR FROM to_timestamp(arrive_time,'YYYY-MM-DD HH24:MI')) AS hour,
  EXTRACT(DOW FROM  to_timestamp(arrive_time,'YYYY-MM-DD HH24:MI')) BETWEEN 1 AND 5 AS weekday,
  COUNT(*) AS ratec
FROM baidu_takeout_rating
GROUP BY arrive_time, hour
ORDER BY arrive_time, hour