set hive.cli.print.header=true;
set hive.cli.print.current.db=true;

create database if not exists article comment 'article information' location '/user/hive/warehouse/article.db/';
--drop table if exists article.article_profile;
-- 创建文章信息表
create table if not exists article.article_data (
    article_id      bigint  comment "article_id",
    channel_id      int     comment "channel_id",
    channel_name    string  comment "channel_name",
    title           string  comment "title",
    content         string  comment "content",
    sentence        string  comment "sentence"
)
    comment "toutiao news_channel"
    location "/root/apache-hive-2.3.7-bin/warehouse/article.db/article_data";
-- 灌数据
load data local inpath "/opt/backup/article.db/article_data" overwrite into table article.article_data;
-- 创建关键词索引信息表
create table if not exists article.idf_keywords_values (
    keyword string  carticle_profileomment "keyword",
    idf     double  comment "idf",
    index   int     comment "index"
)
    comment "toutiao keywords_idf_values"
    location "/root/apache-hive-2.3.7-bin/warehouse/article.db/idf_keywords_values";
-- 创建关键词TF-IDF权重信息表
create table if not exists article.tfidf_keywords_values (
    article_id  bigint  comment "article_id",
    channel_id  int     comment "channel_id",
    keyword     string  comment "keyword",
    tfidf       double  comment "tfidf"
)
    comment "toutiao news_idf_values"
    location "/root/apache-hive-2.3.7-bin/warehouse/article.db/tfidf_keywords_values";
-- 创建关键词TextRank权重信息表
create table if not exists article.textrank_keywords_values (
    article_id  bigint  comment "article_id",
    channel_id  int     comment "channel_id",
    keyword     string  comment "keyword",
    textrank    double  comment "textrank"
)
    comment "toutiao news_textrank_values"
    location "/root/apache-hive-2.3.7-bin/warehouse/article.db/textrank_keywords_values";
-- 创建文章画像信息表
create table if not exists article.article_profile (
    article_id  bigint              comment "article_id",
    channel_id  int                 comment "channel_id",
    keyword     map<string, double> comment "keyword",
    topics      array<string>       comment "topics"
)
    comment "toutiao article_profile"
    location "/root/apache-hive-2.3.7-bin/warehouse/article.db/article_profile";
-- 创建文章向量表
create table if not exists article.article_vector
(
    article_id      bigint          comment "article_id",
    channel_id      int             comment "channel_id",
    articlevector   array<double>   comment "avg_keyword_vector"
);

select count(1) from article_profile where channel_id=18;


create database if not exists profile comment "user action" location '/root/apache-hive-2.3.7-bin/warehouse/profile.db/';
-- 用户行为原始聚合表
create table if not exists profile.user_action
(
    actionTime  string              comment "user actions time",
    readTime    string              comment "user reading time",
    channelId   int                 comment "article channel id",
    param       map<string, string> comment "action parameter"
)
    comment "user primitive action"
    partitioned by (dt string)
    row format serde 'org.apache.hive.hcatalog.data.JsonSerDe'
    location '/root/apache-hive-2.3.7-bin/warehouse/profile.db/user_action';
-- 灌数据
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/{}/' OVERWRITE INTO TABLE user_action PARTITION(dt='{}').format(a, b);

--load data local inpath "/opt/backup/profile.db/user_action/2019-03-05/" overwrite into table profile.user_action partition(dt='20190305');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-05/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190305');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-06/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190306');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-07/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190307');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-08/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190308');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-09/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190309');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-11/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190311');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-12/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190312');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-13/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190313');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-14/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190314');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-15/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190315');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-16/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190316');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-17/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190317');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-18/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190318');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-19/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190319');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-20/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190320');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-21/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190321');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-22/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190322');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-23/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190323');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-24/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190324');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-25/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190325');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-26/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190326');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-27/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190327');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-28/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190328');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-29/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190329');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-03-30/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190330');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-01/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190401');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-02/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190402');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-03/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190403');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-04/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190404');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-05/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190405');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-06/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190406');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-07/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190407');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-08/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190408');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-09/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190409');
LOAD DATA LOCAL INPATH '/opt/backup/profile.db/user_action/2019-04-10/' OVERWRITE INTO TABLE user_action PARTITION(dt='20190410');

-- 用户行为分区表，每条记录对应一条数据
create table if not exists profile.user_article_advanced
(
    user_id     bigint  comment "userID",
    action_time string  comment "user actions time",
    article_id  bigint  comment "articleid",
    channel_id  int     comment "channel_id",
    shared      boolean comment "is shared",
    clicked     boolean comment "is clicked",
    collected   boolean comment "is collected",
    exposure    boolean comment "is exposured",
    read_time   string  comment "reading time"
)
    comment "user primitive action"
    partitioned by (dt string)
    location '/root/apache-hive-2.3.7-bin/warehouse/profile.db/user_article_advanced';

ALTER TABLE user_action DROP IF EXISTS PARTITION(dt='20190410');

-- jar包引起的错误：https://www.cnblogs.com/lfm601508022/p/9188819.html
-- user_action表的规范数据格式：{"actionTime":"2019-04-10 21:04:39","readTime":"","channelId":18,"param":{"action": "click", "userId": "2", "articleId": "14299", "algorithmCombine": "C2"}}
set hive.exec.dynamic.partition.mode=nonstrict;
INSERT OVERWRITE TABLE profile.user_article_advanced PARTITION(dt)
SELECT param['userId'] as user_id
    ,actionTime as action_time
    ,t.article_id
    ,channelId as channel_id
    ,case when param['action'] = 'share' then true else false end as shared
    ,case when param['action'] = 'click' then true else false end as clicked
    ,case when param['action'] = 'collect' then true else false end as collected
    ,true as exposure
    ,readTime as read_time
    ,dt
FROM profile.user_action
LATERAL VIEW
EXPLODE(SPLIT(substring(param['articleId'], 2, length(param['articleId']) - 2), ', ')) t as article_id;

-- 用户行为聚合表
create table if not exists profile.user_article_basic
(
    user_id     bigint  comment "user_id",
    action_time string  comment "user actions time",
    article_id  bigint  comment "article_id",
    channel_id  int     comment "channel_id",
    shared      boolean comment "is shared",
    clicked     boolean comment "is clicked",
    collected   boolean comment "is collected",
    exposure    boolean comment "is exposured",
    read_time   string  comment "reading time"
)
    comment "user_article_basic"
    stored as textfile
    location '/root/apache-hive-2.3.7-bin/warehouse/profile.db/user_article_basic';
-- 灌数据
--load data local inpath "/opt/backup/profile.db/user_article_basic" overwrite into table profile.user_article_basic;
INSERT OVERWRITE TABLE user_article_basic
SELECT user_id
    ,MAX(action_time)
    ,article_id
    ,MAX(channel_id)
    ,MAX(shared)
    ,MAX(clicked)
    ,MAX(collected)
    ,MAX(exposure)
    ,MAX(read_time)
FROM user_article_advanced
group by user_id, article_id
;


-- ********** 用户画像 **********
-- 1.曝光、点击量
select exposure
       ,clicked
       ,count(*)
from user_article_basic
group by exposure, clicked
;
-- 2.区分item、category等，看看点击率
select article_id
       ,count(*) as exposure_cnt
       ,sum(case when clicked = true then 1 else 0 end) / count(*) as ctr
from user_article_basic
group by article_id
;
-- 3.用户基础特征：身高、年龄、性别等
-- 4.用户场景特征：用户对具体item是否点击过等 ==> 拿到用户点击序列等
-- 曝光序列
select user_id
       ,concat_ws(',',collect_set(string(article_id)))
from user_article_basic
group by user_id
;
-- 点击序列
select user_id
       ,concat_ws(',',collect_set(string(article_id)))
from user_article_basic
where clicked = true
group by user_id
;
-- 5.u2i的特征
select user_id
       ,article_id
       ,max(case when clicked = true then 1 else 0 end) as click_or_not
from user_article_basic
group by user_id, article_id
;
-- 6.u2x的特征（开脑洞创造）

