# import findspark
# findspark.init()
from pyspark import SparkConf
from pyspark.sql import SparkSession

# 构建SparkSession
class SparkSessionBase(object):

    SPARK_APP_NAME = None
    # SPARK_URL = "yarn"

    SPARK_EXECUTOR_MEMORY = "16g"
    SPARK_EXECUTOR_CORES = 6
    SPARK_EXECUTOR_INSTANCES = 6

    ENABLE_HIVE_SUPPORT = False

    def _create_spark_session(self):
        conf = SparkConf()

        config = (
            ("spark.app.name", self.SPARK_APP_NAME),
            ("spark.executor.memory", self.SPARK_EXECUTOR_MEMORY),
            # ("spark.master", self.SPARK_URL),
            ("spark.executor.cores", self.SPARK_EXECUTOR_CORES),
            ("spark.executor.instances", self.SPARK_EXECUTOR_INSTANCES),
            # ("spark.sql.warehouse.dir", "/root/apache-hive-2.3.7-bin/warehouse"),
            ("hive.metastore.uris", "thrift://172.18.0.2:9083")
        )

        conf.setAll(config)
        print(self.ENABLE_HIVE_SUPPORT, config)

        if self.ENABLE_HIVE_SUPPORT:
            return SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
        else:
            return SparkSession.builder.config(conf=conf).getOrCreate()
