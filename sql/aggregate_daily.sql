CREATE TABLE baidu_rates_daily_aq AS
SELECT
  rt.day,
  rt.weekday,
  rt.ratec,
  aq.avg_aqi,
  aq.avg_pm10,
  aq.avg_pm2_5
FROM baidu_daily_rates as rt
LEFT JOIN bj_aq_avg_daily as aq
ON rt.day = aq.day
AND aq.station_code = '1006A'
WHERE rt.day > '2014-12-31'
ORDER BY rt.day