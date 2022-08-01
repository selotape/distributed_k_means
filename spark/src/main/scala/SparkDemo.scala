import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object SparkDemo {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.INFO)
    val sc = new SparkContext("local[*]", "SparkDemo")
    val lines = sc.textFile("../README.md")
    val words = lines.flatMap(line => line.split(' '))
    val words_lower = words.map(x => x.toLowerCase())
    val wordsKVRdd = words_lower.map(x => (x, 1))
    val count = wordsKVRdd
      .reduceByKey((x, y) => x + y)
      .map(x => (x._2, x._1))
      .sortByKey(ascending = false)
      .map(x => (x._2, x._1))
      .take(100)
    count.foreach(println)
  }
}