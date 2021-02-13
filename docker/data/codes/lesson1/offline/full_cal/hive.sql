create database if not exists toutiao comment "user,news information of mysql" location '/root/apache-hive-2.3.7-bin/warehouse/toutiao.db/';
-- 用户信息表
create table user_profile
(
    user_id             BIGINT comment "userid",
    gender              BOOLEAN comment "gender",
    birthday            STRING comment "birthday",
    real_name           STRING comment "real_name",
    create_time         STRING comment "create_time",
    update_time         STRING comment "update_time",
    register_media_time STRING comment "register_media_time",
    id_number           STRING comment "id_number",
    id_card_front       STRING comment "id_card_front",
    id_card_back        STRING comment "id_card_back",
    id_card_handheld    STRING comment "id_card_handheld",
    area                STRING comment "area",
    company             STRING comment "company",
    career              STRING comment "career"
)
    COMMENT "toutiao user profile"
    row format delimited fields terminated by ','
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/toutiao.db/user_profile';
-- 用户基础信息表
create table user_basic
(
    user_id         BIGINT comment "user_id",
    mobile          STRING comment "mobile",
    password        STRING comment "password",
    profile_photo   STRING comment "profile_photo",
    last_login      STRING comment "last_login",
    is_media        BOOLEAN comment "is_media",
    article_count   BIGINT comment "article_count",
    following_count BIGINT comment "following_count",
    fans_count      BIGINT comment "fans_count",
    like_count      BIGINT comment "like_count",
    read_count      BIGINT comment "read_count",
    introduction    STRING comment "introduction",
    certificate     STRING comment "certificate",
    is_verified     BOOLEAN comment "is_verified"
)
    COMMENT "toutiao user basic"
    row format delimited fields terminated by ','
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/toutiao.db/user_basic';
-- 文章基础信息表
create table news_article_basic
(
    article_id  BIGINT comment "article_id",
    user_id     BIGINT comment "user_id",
    channel_id  BIGINT comment "channel_id",
    title       STRING comment "title",
    status      BIGINT comment "status",
    update_time STRING comment "update_time"
)
    COMMENT "toutiao news_article_basic"
    row format delimited fields terminated by ','
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/toutiao.db/news_article_basic';
-- 文章频道表
create table news_channel
(
    channel_id   BIGINT comment "channel_id",
    channel_name STRING comment "channel_name",
    create_time  STRING comment "create_time",
    update_time  STRING comment "update_time",
    sequence     BIGINT comment "sequence",
    is_visible   BOOLEAN comment "is_visible",
    is_default   BOOLEAN comment "is_default"
)
    COMMENT "toutiao news_channel"
    row format delimited fields terminated by ','
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/toutiao.db/news_channel';
-- 文章内容表
create table news_article_content
(
    article_id BIGINT comment "article_id",
    content    STRING comment "content"
)
    COMMENT "toutiao news_article_content"
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/toutiao.db/news_article_content';

-- array=(user_profile user_basic news_article_basic news_channel news_article_content)

-- for table_name in ${array[@]};
-- do
--     sqoop import \
--         --connect jdbc:mysql://172.18.0.5:3306/toutiao \
--         --username root \
--         --password root \
--         --table $table_name \
--         --m 5 \
--         --hive-home apache-hive-2.3.7-bin \
--         --hive-import \
--         --create-hive-table  \
--         --hive-drop-import-delims \
--         --warehouse-dir /root/apache-hive-2.3.7-bin/warehouse/toutiao.db \
--         --hive-table toutiao.$table_name 
-- done

LOAD DATA LOCAL INPATH '/opt/backup/toutiao.db/user_profile' OVERWRITE INTO TABLE user_profile;
LOAD DATA LOCAL INPATH '/opt/backup/toutiao.db/user_basic' OVERWRITE INTO TABLE user_basic;
LOAD DATA LOCAL INPATH '/opt/backup/toutiao.db/news_article_basic' OVERWRITE INTO TABLE news_article_basic;
LOAD DATA LOCAL INPATH '/opt/backup/toutiao.db/news_channel' OVERWRITE INTO TABLE news_channel;
LOAD DATA LOCAL INPATH '/opt/backup/toutiao.db/news_article_content' OVERWRITE INTO TABLE news_article_content;

-- hive -e

set hive.cli.print.header=true;
select * from user_profile;



create database if not exists profile comment "use action" location '/root/apache-hive-2.3.7-bin/warehouse/profile.db/';
create table user_action
(
    actionTime STRING comment "user actions time",
    readTime   STRING comment "user reading time",
    channelId  INT comment "article channel id",
    param MAP<STRING, STRING> comment "action parameter"
)
    COMMENT "user primitive action"
    PARTITIONED BY (dt STRING)
    ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/profile.db/user_action';

LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-05/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190305');


create database if not exists article comment 'artcile information' location '/user/hive/warehouse/article.db/';
-- 创建文章信息表
CREATE TABLE article_data
(
    article_id   BIGINT comment "article_id",
    channel_id   INT comment "channel_id",
    channel_name STRING comment "channel_name",
    title        STRING comment "title",
    content      STRING comment "content",
    sentence     STRING comment "sentence"
)
    COMMENT "toutiao news_channel";
    -- LOCATION '/root/apache-hive-2.3.7-bin/warehouse/article.db/article_data';
-- 创建关键词索引信息表
CREATE TABLE idf_keywords_values
(
    keyword STRING comment "article_id",
    idf     DOUBLE comment "idf",
    index   INT comment "index"
);
-- 创建关键词TF-IDF权重信息表
CREATE TABLE tfidf_keywords_values
(
    article_id INT comment "article_id",
    channel_id INT comment "channel_id",
    keyword    STRING comment "keyword",
    tfidf      DOUBLE comment "tfidf"
);
-- 创建关键词TextRank权重信息表
CREATE TABLE textrank_keywords_values
(
    article_id INT comment "article_id",
    channel_id INT comment "channel_id",
    keyword    STRING comment "keyword",
    textrank   DOUBLE comment "textrank"
);
-- 创建文章画像信息表
CREATE TABLE article_profile
(
    article_id INT comment "article_id",
    channel_id INT comment "channel_id",
    keywords    MAP<STRING, DOUBLE> comment "keyword",
    topics     ARRAY<STRING> comment "topics"
); 

CREATE TABLE article_vector
(
    article_id INT comment "article_id",
    channel_id INT comment "channel_id",
    articlevector ARRAY<DOUBLE> comment "keyword"
);

LOAD DATA LOCAL INPATH '/opt/backup/article.db/article_profile' OVERWRITE INTO TABLE article_profile;

LOAD DATA LOCAL INPATH '/opt/backup/article.db/article_vector' OVERWRITE INTO TABLE article_vector;


create table user_article_basic
(
    user_id     BIGINT comment "userID",
    action_time STRING comment "user actions time",
    article_id  BIGINT comment "articleid",
    channel_id  INT comment "channel_id",
    shared      BOOLEAN comment "is shared",
    clicked     BOOLEAN comment "is clicked",
    collected   BOOLEAN comment "is collected",
    exposure    BOOLEAN comment "is exposured",
    read_time   STRING comment "reading time"
)
    COMMENT "user_article_basic"
    STORED as textfile
    LOCATION '/root/apache-hive-2.3.7-bin/warehouse/profile.db/user_article_basic';

LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_article_basic' OVERWRITE INTO TABLE user_article_basic;



"""
curl -H 'Content-Type: application/json' -XPUT 'localhost:9200/article_vector' -d '
{
  "settings": {
    "index": {
      "number_of_replicas": "1", 
      "number_of_shards": "5"
    }
  },
  "mappings": {
    "properties": {
        "article_id": {
            "type": "integer"
        },
        "channel_id": {
            "type": "integer"
        },
        "articlevector": {
            "type": "dense_vector",
            "dims": 100
        }
    }
  }
}'

curl -H 'Content-Type: application/json' -XPUT 'localhost:9200/article_vector/_doc/1?pretty' -d '
{
    "article_id": 123,
    "channel_id" : 456,
    "articlevector": [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
}'

curl -XGET 'localhost:9200/article_vector/_search?pretty'

curl -H 'Content-Type: application/json' -XGET 'localhost:9200/article_vector/_search?pretty' -d '
{
  "size": 2,
  "query": 
    {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, \u0027articlevector\u0027) + 1.0",
                "params": {
                    "query_vector": [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
                }
            }
        }
    }
}
'
"""

select user_id
    ,concat_ws(',',collect_set(string(article_id)))
from user_article_basic 
group by user_id
;