SELECT COUNT(*) FROM baidu_takeout_rating;

select pass_uid,shop_id,arrive_time, count(*)
from baidu_takeout_rating
group by pass_uid,shop_id,arrive_time
HAVING count(*) > 1


