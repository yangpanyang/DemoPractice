import java.io.File

import org.apache.spark.sql.{Row, SaveMode, SparkSession}

object Program {
  def main(args: Array[String]) = {
    val spark = SparkSession
        .builder()
        .appName("Spark Hive Example")
        .config("spark.es.nodes", "172.18.0.6:9200")
        .enableHiveSupport()
        .getOrCreate()


    textFile = spark.read.text('LICENSE')
    wordCounts = textFile.rdd.flatMap(lambda line: line.value.split())
    # emit value:1 for each key:word
    wordCounts = wordCounts.map(lambda word: (word, 1))
    # add up word counts by key:word
    wordCounts = wordCounts.reduceByKey(lambda a, b: a+b)
    # sort in descending order by word counts
    wordCounts = wordCounts.sortBy(lambda item: -item[1])
    # collect the results in an array
    results = wordCounts.collect()
    # print the first ten elements
    print(results[:10])
  }
}
