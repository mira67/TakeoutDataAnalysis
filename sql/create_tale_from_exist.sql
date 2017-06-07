CREATE TABLE postgres.baidu_takeout_rating_ex AS
( SELECT
    postgres.baidu_takeout_rating_extend.mycmtid,
    postgres.baidu_takeout_rating_extend.score,
    postgres.baidu_takeout_rating_extend.cal_score,
    postgres.baidu_takeout_rating_extend.sfrom,
    postgres.baidu_takeout_rating_extend.pass_uid,
    postgres.baidu_takeout_rating_extend.create_time,
    postgres.baidu_takeout_rating_extend.audit_time,
    postgres.baidu_takeout_rating_extend.cost_time,
    postgres.baidu_takeout_rating_extend.arrive_time,
    postgres.baidu_takeout_rating_extend.pass_name,
    postgres.baidu_takeout_rating_extend.service_score,
    postgres.baidu_takeout_rating_extend.dish_score,
    postgres.baidu_takeout_rating_extend.is_anonymous,
    postgres.baidu_takeout_rating_extend.city_id,
    postgres.baidu_takeout_rating_extend.product_scheme,
    postgres.baidu_takeout_rating_extend.shop_id,
 	EXTRACT(HOUR FROM to_timestamp(postgres.baidu_takeout_rating_extend.arrive_time,'YYYY-MM-DD HH24:MI')) AS hourofday,
  	EXTRACT(DOW FROM to_timestamp(postgres.baidu_takeout_rating_extend.arrive_time,'YYYY-MM-DD HH24:MI')) AS dayofweek
 FROM postgres.baidu_takeout_rating_extend
);

CREATE INDEX rating_id
    ON postgres.baidu_takeout_rating_ex USING btree
    (pass_uid COLLATE pg_catalog."default", arrive_time COLLATE pg_catalog."default", cost_time COLLATE pg_catalog."default")
    TABLESPACE pg_default;