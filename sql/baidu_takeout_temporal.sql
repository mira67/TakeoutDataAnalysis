SELECT 
    shop_id,
    sum(case when dayOfweek between 1 and 5 then 1 else 0 end) as weekday,
    sum(case when dayOfweek = 6 then 1 else 0 end) as saturday,
    sum(case when dayOfweek = 0 then 1 else 0 end) as sunday,
    sum(case when hourOfday = 0 then 1 else 0 end) as h00,
	sum(case when hourOfday = 1 then 1 else 0 end) as h01,
    sum(case when hourOfday = 2 then 1 else 0 end) as h02,
    sum(case when hourOfday = 3 then 1 else 0 end) as h03,
    sum(case when hourOfday = 4 then 1 else 0 end) as h04,
	sum(case when hourOfday = 5 then 1 else 0 end) as h05,
    sum(case when hourOfday = 6 then 1 else 0 end) as h06,
    sum(case when hourOfday = 7 then 1 else 0 end) as h07,
    sum(case when hourOfday = 8 then 1 else 0 end) as h08,
	sum(case when hourOfday = 9 then 1 else 0 end) as h09,
    sum(case when hourOfday = 10 then 1 else 0 end) as h10,
    sum(case when hourOfday = 11 then 1 else 0 end) as h11,
    sum(case when hourOfday = 12 then 1 else 0 end) as h12,
	sum(case when hourOfday = 13 then 1 else 0 end) as h13,
    sum(case when hourOfday = 14 then 1 else 0 end) as h14,
    sum(case when hourOfday = 15 then 1 else 0 end) as h15,
    sum(case when hourOfday = 16 then 1 else 0 end) as h16,
    sum(case when hourOfday = 17 then 1 else 0 end) as h17,
	sum(case when hourOfday = 18 then 1 else 0 end) as h18,
    sum(case when hourOfday = 19 then 1 else 0 end) as h19,
    sum(case when hourOfday = 20 then 1 else 0 end) as h20,
    sum(case when hourOfday = 21 then 1 else 0 end) as h21,
	sum(case when hourOfday = 22 then 1 else 0 end) as h22,
    sum(case when hourOfday = 23 then 1 else 0 end) as h23
FROM
    baidu_takeout_temporal
WHERE pass_uid = '3337871'
GROUP BY 
    shop_id
ORDER BY shop_id