CREATE TABLE bj_eleme_shop_rates AS
SELECT
  rated_date,
  id,
  EXTRACT(DOW FROM rated_date) BETWEEN 1 AND 5 AS weekday,
  COUNT(*) AS rates_num
FROM bj_eleme_all_rate
GROUP BY id,rated_date;
