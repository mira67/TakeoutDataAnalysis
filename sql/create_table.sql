CREATE TABLE bj_aq (
  area varchar DEFAULT NULL,
  position_name varchar DEFAULT NULL,
  station_code varchar DEFAULT NULL,
  time_point timestamp DEFAULT NULL,
  quality varchar DEFAULT NULL,
  aqi varchar DEFAULT NULL,
  pm10 numeric DEFAULT NULL,
  pm10_24h numeric DEFAULT NULL,
  pm2_5 numeric DEFAULT NULL,
  pm2_5_24h numeric DEFAULT NULL,
  date_time timestamp DEFAULT NULL, 
  day timestamp DEFAULT NULL,
  hour integer DEFAULT NULL
);

COPY bj_aq FROM 'C:\Users\AA_LQ\Downloads\BJ_pm2.5_pm10_aqi_201501_201702.csv' CSV HEADER;


