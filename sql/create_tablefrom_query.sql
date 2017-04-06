CREATE TABLE bj_eleme_shops_rate AS
(SELECT 
 bj_eleme_shop_rates.id,
 bj_eleme_shop_rates.rated_date,
 bj_eleme_shop_rates.weekday,
 bj_eleme_shop_rates.rates_num,
 bj_eleme_all_shop.latitude,
 bj_eleme_all_shop.longitude
 FROM bj_eleme_shop_rates
 LEFT JOIN bj_eleme_all_shop
 ON bj_eleme_shop_rates.id = bj_eleme_all_shop.id
 ORDER BY  bj_eleme_shop_rates.id, bj_eleme_shop_rates.rated_date
);