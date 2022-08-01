package org.ronvis.soccer.demo

import org.apache.spark.ml.clustering.MyKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.ml.feature.VectorAssembler

object SoccerDemo {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("SOCCER example")
      .getOrCreate()

    //    val dataset = spark.read.format("libsvm").load("../datasets/sample_kmeans_data.txt")
    val dataset = getKddCup99Dataset(spark)

    //     Trains a k-means model.
    val kmeans = new MyKMeans().setK(10).setSeed(1L)
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

  def getSampleKMeansDataset(spark: SparkSession): Dataset[_] = {
    spark.read.format("libsvm").load("../datasets/sample_kmeans_data.txt")
  }

  def getKddCup99Dataset(spark: SparkSession): Dataset[_] = {
    // TODO - change to "drop all none numeric columns"
    val frame = spark.read.format("csv").option("inferSchema", "true").load("../datasets/kddcup99/kddcup.data").drop("_c1", "_c2", "_c3", "_c41")
    val assembler = new VectorAssembler().setInputCols(frame.columns).setOutputCol("features")
    assembler.transform(frame)
  }
}