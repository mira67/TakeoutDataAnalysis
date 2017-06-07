CREATE INDEX index_id
    ON postgres.baidu_takeout_rating_extend USING btree
    (pass_uid COLLATE pg_catalog."default", arrive_time COLLATE pg_catalog."default", cost_time COLLATE pg_catalog."default")
    TABLESPACE pg_default;