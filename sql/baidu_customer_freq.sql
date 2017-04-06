CREATE table baidu_takeout_customer AS (
SELECT 
pass_uid,
count(*) as user_freq
FROM
baidu_takeout_rating
GROUP BY
pass_uid
ORDER BY user_freq,pass_uid
)