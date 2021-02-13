# 通过image创建一个container，并起container服务，然后进入container
sudo docker run -itd --name containerName --hostname dockerHostName imzhen/doc_project_docker:20201122 &> /dev/null  # [/bin/bash]
sudo docker exec -it containerName bash

# get into hadoop master container
sudo docker exec -it hadoop-master bash

# hadoop
sudo docker pull kiwenlau/hadoop:1.0
sudo docker network create --driver=bridge hadoop
git clone https://github.com/kiwenlau/hadoop-cluster-docker

./start-hadoop.sh  # 到docker上把服务起起来
# 本地访问hadoop集群
# NameNode: http://127.0.0.1:50070/
# ResourceManager: http://127.0.0.1:8088/

# mysql
docker pull mysql:5.6
docker run -itd --net=hadoop --hostname hadoop-mysql --name hadoop-mysql -p 3308:3306 -e MYSQL_ROOT_PASSWORD=root -v /Users/yangpan\ 1/works/docker/data:/var/lib/mysql mysql:5.6
mysql -uroot -proot -P 3308 -h 127.0.0.1  # 进入本地的mysql（安装链接：https://www.jianshu.com/p/07a9826898c0）
# create database hive;

# 之后通过 docker inspect 检查该容器的ip，我获取到的ip是172.18.0.5
# docker inspect hadoop-mysql | grep "IPAddress"
# docker inspect hadoop-master | grep "IPAddress"


# hive，配置到docker上
wget http://mirror.bit.edu.cn/apache/hive/hive-2.3.7/apache-hive-2.3.7-bin.tar.gz
tar -zxvf apache-hive-2.3.7-bin.tar.gz
# hive的源数据在mysql里面，hive就需要通过这个jar包与mysql沟通，把jar包放到lib里面去
wget -q "http://search.maven.org/remotecontent?filepath=mysql/mysql-connector-java/5.1.38/mysql-connector-java-5.1.38.jar" -O mysql-connector-java.jar
cd apache-hive-2.3.7-bin
cp ../mysql-connector-java.jar lib
# 配置怎么和mysql进行交互
cp conf/hive-default.xml.template conf/hive-site.xml

# 一些源信息的配置
<property>
    <name>system:java.io.tmpdir</name>
    <value>/tmp/hive/java</value>
</property>
<property>
    <name>system:user.name</name>
    <value>${user.name}</value>
</property>

# 和mysql交互的user、password
<property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>root</value>
</property>
<property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>root</value>
</property>

# 把ConnectionURL写进去，172.18.0.5是mysql的docker网址（通过docker inspect找到），需要改
<property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://172.18.0.5:3306/hive?createDatabaseIfNotExist=true&amp;useSSL=false</value>
</property>
<property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.jdbc.Driver</value>
</property>

# 把hive的启动命令加进去，每次重启docker后需要进行source /etc/profile命令，否则找不到hive命令，/etc/profile系统初始化的时候需要做的一些事情
echo 'export HIVE_HOME="/root/apache-hive-2.3.7-bin"\nexport PATH=$PATH:$HIVE_HOME/bin'>> /etc/profile && source /etc/profile
# 把hive的初始环境给init出来，会出现各种各样的报错，stackoverflow都可以解决
schematool -initSchema -dbType mysql

# hive命令
show databases;
use article;
select * from article.article_data;
select count(*) from article_data;
# Tracking URL暂时打不开，可能由于端口没有被暴露出来


# # sqoop
# wget http://mirror.bit.edu.cn/apache/sqoop/1.4.7/sqoop-1.4.7.bin__hadoop-2.6.0.tar.gz && tar -zxvf sqoop-1.4.7.bin__hadoop-2.6.0.tar.gz
# echo 'export SQOOP_HOME=/root/sqoop-1.4.7.bin__hadoop-2.6.0\nexport PATH=$PATH:$SQOOP_HOME/bin' >> /etc/profile && source /etc/profile

# cd /root/sqoop-1.4.7.bin__hadoop-2.6.0/conf && cp sqoop-env-template.sh sqoop-env.sh && echo 'export HADOOP_COMMON_HOME=$HADOOP_HOME\nexport HADOOP_MAPRED_HOME=$HADOOP_HOME\nexport HIVE_HOME=$HIVE_HOME' >> sqoop-env.sh && source /etc/profile
# cp /root/mysql-connector-java.jar /root/sqoop-1.4.7.bin__hadoop-2.6.0/lib

# sqoop list-databases --connect jdbc:mysql://172.18.0.5:3306 --username root --password 'root'

# # flume
# wget https://mirror.bit.edu.cn/apache/flume/1.7.0/apache-flume-1.7.0-bin.tar.gz && tar -zxvf apache-flume-1.7.0-bin.tar.gz
# echo 'export FLUME_HOME=/root/apache-flume-1.7.0-bin\nexport PATH=$PATH:$FLUME_HOME/bin' >> /etc/profile && source /etc/profile


# 由于spark版本比较高，Java版本比较低，所以需要先把Java8升级出来
# sudo apt-get install python-software-properties
# sudo apt-get update
# sudo apt install software-properties-common
# sudo apt-get update
# sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk
sudo update-alternatives --config java
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> /etc/profile && source /etc/profile
echo $JAVA_HOME  #版本打印出来看看

# spark安装
# Hadoop架构：worker + master
# Spark架构：driver + executor / driver master + worker
wget https://mirror.bit.edu.cn/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz && tar -zxvf spark-2.4.7-bin-hadoop2.7.tgz
# 跑wordcount，应该是看看能不能正常运行
/root/spark-2.4.7-bin-hadoop2.7/bin/spark-submit --class \
    org.apache.spark.examples.SparkPi \
    /root/spark-2.4.7-bin-hadoop2.7/examples/jars/spark-examples_2.11-2.4.7.jar 1000

# spark examples，环境搭建，单机版本在本docker上跑
/root/spark-2.4.7-bin-hadoop2.7/bin/spark-class org.apache.spark.deploy.master.Master --ip `hostname` --port 7077 --webui-port 10000
# 集群版本，worker需要在其它hadoop slave上跑
# /root/spark-2.4.7-bin-hadoop2.7/bin/spark-class org.apache.spark.deploy.worker.Worker --ip `hostname` --port 7077 --webui-port 10000

hive -S -e "describe formatted article.article_data ;" | grep 'Location' | awk '{ print $NF }'
# 数据存储在hadoop上的地址：hdfs://hadoop-master:9000/root/apache-hive-2.3.7-bin/warehouse/article.db/article_data
hive -S -e "show databases;"
# 至此，hadoop、spark环境都有了，spark环境需要访问hive集群，需要知道hive的源数据在哪儿，才知道怎么去读这个表，有2种读的方式：
# （1）hive本身源数据在mysql，spark可以像hive一样去读mysql，类似hadoop的沟通方式把mysql-connector-java.jar包放置spark/jar目录下即可
# （2）通过hive启动一个本地的metastore服务，实现hive与spark交互
# 把以下配置同样加入hive/conf/hive-site.xml文件内
# hive.metastore.uris：让hive自己起一个服务，这个服务是一个远程的metastore，这样其它引擎就可以连接到这个服务，读取hive的源数据
vi conf/hive-site.xml
"""
<property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse</value>  # 改为：hdfs://hadoop-master:9000/root/apache-hive-2.3.7-bin/warehouse
</property>
<property>
    <name>hive.metastore.local</name>
    <value>false</value>
</property>
<property>
    <name>hive.metastore.uris</name>
    <value>thrift://172.18.0.2:9083</value>
</property>
"""
# 后台启动这个服务，默认是不起的；但是退出后任务还在
nohup hive --service metastore 2>&1 >> /var/log.log &
netstat -tpnl | grep 9083
sbin/start-all.sh  # 启动spark集群
./spark-shell --master spark://hadoop-master:7077  # 启动spark-shell访问hive上数据
spark.sql("show tables").show  # 查询hive
hdfs dfsadmin -safemode leave 关闭安全模式即可

# 同样把配置好的文件放到spark里面
cp conf/hive-site.xml /root/spark-2.4.7-bin-hadoop2.7/conf


# python配置
# 需要先加载数据源、再update，才能识别出python3.6
# apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo add-apt-repository ppa:deadsnakes/ppa  # 一般都会出错
sudo apt-get update
# 升级python
apt-get install python3.6 python3.6-dev
ln -fs /usr/bin/python3.6 /usr/bin/python3  # 建立软链接
apt-get install python3-pip
# 升级pip3
# pip3 install --upgrade pip
# python3 -m pip install --upgrade pip
# python3 -m pip install jupyter --no-cache-dir # 下面不行就得这样装
pip install jupyter --no-cache-dir  # 能正常安装就没问题，不行就跑下下面这2条命令
wget https://bootstrap.pypa.io/ez_setup.py -O - | python3
# pip install -U --no-use-wheel pip


# how to run code in docker by vscode
# 做远程开发，推荐使用VSCode，安装remote-containers插件，这样就可以在远端执行

# HDFS常用命令
hdfs dfs -ls hdfs://hadoop-master:10000/root/
hdfs dfs -chmod -R 777 /tmp


# ES配置
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.10.0  # 官方docker
docker run -itd -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" --hostname hadoop-es --name=hadoop-es --network=hadoop docker.elastic.co/elasticsearch/elasticsearch:7.10.0
sudo docker exec -it hadoop-es bash
ps aux  # 进入docker执行，可以看到es已经开了，不需要额外的操作

curl '127.0.0.1:9200'  # run服务起来之后，直接在本地就可以curl
# 看看ES有些啥
curl -XGET "localhost:9200/_cat/health?v"  # 本机健康程度
curl -XGET "localhost:9200/_cat/indices?v"  # 查看集群信息
# 生成索引
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
curl -XGET "localhost:9200/article_vector/_mapping"  # 看看构建出来的索引长什么样子
curl -XDELETE "localhost:9200/article_vector"  # 删除索引
# 插入数据，pretty让结果显示的漂亮一些
curl -H 'Content-Type: application/json' -XPUT 'localhost:9200/article_vector/_doc/1/?pretty' -d '
{
    "article_id": 123,
    "channel_id": 456,
    "articlevector": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
}'

# 老师的命令，应该是test用的
# wget https://artifacts.elastic.co/downloads/elasticsearch-hadoop/elasticsearch-hadoop-7.10.0.zip
# unzip elasticsearch-hadoop-7.10.0.zip
# hdfs dfs -mkdir /user/test/es_hadoop/
# hdfs dfs -put elasticsearch-hadoop-hive-7.10.0.jar /user/test/es_hadoop/
# ADD JAR hdfs://test/user/test/es_hadoop/elasticsearch-hadoop-hive-7.10.0.jar;

# 在Hadoop的主机群上启动，实现es、hive、spark的连接
./bin/spark-shell --jars jars/elasticsearch-hadoop-7.10.0.jar --conf spark.es.nodes="172.18.0.6:9200" spark.sql.catalogImplementation="hive" hive.metastore.uris="thrift://172.18.0.2:9083"
# --jars
# jars/elasticsearch-hadoop-7.10.0.jar  # 连接的jar包
# --conf
# spark.es.nodes="172.18.0.6:9200"  # 对应的nodes在哪儿，用以找到es
# spark.sql.catalogImplementation="hive"  # 实现enableHiveSupport
# hive.metastore.uris="thrift://172.18.0.2:9083"  # hive的metastore服务的位置

# 进入Scala，导入环境
# spark原生有SQLContext，就是在python里面跑的那些，但是要把数据从hive导入到es，就需要用es官方提供的SQLContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLContext._
import org.elasticsearch.spark.sql._

# 把数据导入到es
val article = sql("select * from article.article_vector limit 1000")
article.saveToEs("article_vector")

# 全局搜索
curl -XGET 'localhost:9200/article_vector/_search?pretty'
# 近邻搜索
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
                    "query_vector": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
                }
            }
        }
    }
}'
# size  # 返回2个结果
# match_all  # 把所有的结果都拿过来计算
# source  # 计算余弦相似度，\u0027代表单引号的unicode，es结果不允许小于0所以要+1.0


# 分布式zookeeper搭建
# ssh
# hadoop-slave1
ssh-keygen -t rsa  # 生成密钥
cp ~/.ssh/id_rsa.pub ~/.ssh/hadoop_slave1_id_rsa.pub
scp ~/.ssh/hadoop_slave1_id_rsa.pub hadoop-master:~/.ssh/  # 拷贝至master
# hadoop-slave2
ssh-keygen -t rsa
cp ~/.ssh/id_rsa.pub ~/.ssh/hadoop_slave2_id_rsa.pub
scp ~/.ssh/hadoop_slave2_id_rsa.pub hadoop-master:~/.ssh/
# master
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/hadoop_slave1_id_rsa.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/hadoop_slave2_id_rsa.pub >> ~/.ssh/authorized_kyes  # 得到完整的密钥
# 拷贝文件至slave1及slave2
scp ~/.ssh/authorized_keys hadoop-slave1:~/.ssh
scp ~/.ssh/authorized_keys hadoop-slave2:~/.ssh  # 3台机器同步密钥，以保证互通
# zookeeper配置
wget https://mirror.bit.edu.cn/apache/zookeeper/zookeeper-3.6.2/apache-zookeeper-3.6.2-bin.tar.gz && tar -zxf apache-zookeeper-3.6.2-bin.tar.gz
mkdir apache-zookeeper-3.6.2-bin/data
cp apache-zookeeper-3.6.2-bin/conf/zoo_sample.cfg apache-zookeeper-3.6.2-bin/conf/zoo.cfg
vi apache-zookeeper-3.6.2-bin/conf/zoo.cfg
"""
dataDir=/root/apache-zookeeper-3.6.2-bin/data
# 下面三个是分布式用的
server.1=hadoop-master:2888:3888
server.2=hadoop-slave1:2888:3888
server.3=hadoop-slave2:2888:3888
"""
scp -r apache-zookeeper-3.6.2-bin hadoop-slave1:/root
scp -r apache-zookeeper-3.6.2-bin hadoop-slave2:/root
# for each of three
touch /root/apache-zookeeper-3.6.2-bin/data/myid && echo 1 > /root/apache-zookeeper-3.6.2-bin/data/myid
touch /root/apache-zookeeper-3.6.2-bin/data/myid && echo 2 > /root/apache-zookeeper-3.6.2-bin/data/myid
touch /root/apache-zookeeper-3.6.2-bin/data/myid && echo 3 > /root/apache-zookeeper-3.6.2-bin/data/myid
# for each of three
/root/apache-zookeeper-3.6.2-bin/bin/zkServer.sh start


## HBase配置
wget https://mirror.bit.edu.cn/apache/hbase/2.3.3/hbase-2.3.3-bin.tar.gz && tar -zxf hbase-2.3.3-bin.tar.gz
mkdir hbase-2.3.3/logs
vi hbase-2.3.3/conf/hbase-env.sh
echo -e 'export HBASE_HOME="/root/hbase-2.3.3"\nexport PATH=$PATH:$HBASE_HOME/bin' >> /etc/profile && source /etc/profile
echo "export JAVA_HOME=${JAVA_HOME}" >> hbase-2.3.3/conf/hbase-env.sh
# 这里同时注意 hadoop的env也要改...
vi /usr/local/hadoop/etc/hadoop/hadoop-env.sh
export JAVA_HOME=${JAVA_HOME}  # change 7 to 8
# 如果是单机,HBASE_MANAGES_ZK=true, 下面的regionservers也要改, 以及hbase.zookeeper.quorum
echo -e 'export HBASE_LOG_DIR=${HBASE_HOME}/logs\nexport HBASE_MANAGES_ZK=false\nexport HBASE_PID_DIR=/var/hadoop/pids' >> hbase-2.3.3/conf/hbase-env.sh
rm -rf hbase-2.3.3/conf/regionservers && echo -e "hadoop-master\nhadoop-slave1\nhadoop-slave1" >> ~/hbase-2.3.3/conf/regionservers
vi ~/hbase-2.3.3/conf/hbase-site.xml
"""
<property>
    <name>hbase.rootdir</name>
    <value>hdfs://hadoop-master:9000/hbase</value>
</property>
<property>
    <name>hbase.cluster.distributed</name>
    <value>true</value>
</property>
<property>
    <name>hbase.zookeeper.quorum</name>
    <value>hadoop-master,hadoop-slave1,hadoop-slave2</value>
</property>
<property>
    <name>hbase.zookeeper.property.dataDir</name>
    <value>/root/apache-zookeeper-3.6.2-bin/data</value>
</property>
<property>
    <name>hbase.master</name>
    <value>hdfs://hadoop-master:60000</value>
</property>
"""
scp -r /root/hbase-2.3.3 hadoop-slave1:/root
scp -r /root/hbase-2.3.3 hadoop-slave2:/root
./hbase-2.3.3/bin/start-hbase.sh

# 常用命令
./hbase-2.3.3/bin/start-hbase.sh
hbase shell
status
describe 'user_profile'
