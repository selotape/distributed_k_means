package org.ronvis.soccer.demo

import org.apache.spark.ml.clustering.MyKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession

object SoccerDemo {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("SOCCER example")
//      .config("spark.some.config.option", "some-value")
      .getOrCreate()

//    val dataset = spark.read.format("libsvm").load("../datasets/kddcup99/kddcup.data")
    val dataset = spark.read.format("libsvm").load("../datasets/sample_kmeans_data.txt")

    // Trains a k-means model.
    val kmeans = new MyKMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)

    // Make predictions
    val predictions = model.transform(dataset)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

  }
}