CREATE table baidu_takeout_customer AS (
SELECT 
pass_uid,
count(*) as user_freq,
count(DISTINCT shop_id) as shop_num,
count(DISTINCT date(arrive_time)) as date_num 
FROM
baidu_takeout_rating
WHERE date(arrive_time)
BETWEEN '2016-01-01' AND '2017-03-01'
GROUP BY
pass_uid
ORDER BY user_freq,pass_uid
)