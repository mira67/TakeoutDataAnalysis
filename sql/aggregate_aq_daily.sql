CREATE TABLE bj_aq_avg_daily AS
SELECT
  date(time_point) AS day,
  station_code,
  AVG(to_number(aqi, '9999.99')) AS avg_aqi,
  AVG(pm10) AS avg_pm10,
  AVG(pm10_24h) AS avg_pm10_24h,
  AVG(pm2_5) AS avg_pm2_5,
  AVG(pm2_5_24h) AS avg_pm2_5_24h
FROM bj_aq
WHERE date(time_point) >= '2015-01-01'
GROUP BY station_code,date(time_point)
ORDER BY station_code,date(time_point);
