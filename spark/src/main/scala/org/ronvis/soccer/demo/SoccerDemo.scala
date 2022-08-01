package org.ronvis.soccer.demo

import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.Logger
import org.apache.spark.ml.clustering.{KMeans, MyKMeans}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, SparkSession}

object SoccerDemo {
  lazy val log: Logger = LogManager.getLogger(this.getClass)


  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("SOCCER example")
      .getOrCreate()

    val dataset = getSampleKMeansDataset(spark)
    //    val dataset = getKddCup99Dataset(spark)

    val seed = 1L
    val k = 10
    val myKmeans = new MyKMeans().setK(k).setSeed(seed)
    fitAndEvaluate(myKmeans, dataset)

    val boringOldschoolKmeans = new KMeans().setK(k).setSeed(seed)
    fitAndEvaluate(boringOldschoolKmeans, dataset)
  }

  def fitAndEvaluate(kmeans: Estimator[_ <: Model[_]], dataset: DataFrame): Unit = {
    val model = kmeans.fit(dataset)
    val predictions = model.transform(dataset)
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)
    log.info(s"Silhouette with squared euclidean distance = $silhouette")
  }

  def getSampleKMeansDataset(spark: SparkSession): DataFrame = {
    spark.read.format("libsvm").load("../datasets/sample_kmeans_data.txt")
  }

  def getKddCup99Dataset(spark: SparkSession): DataFrame = {
    // TODO - change to "drop all none numeric columns"
    val frame = spark.read.format("csv").option("inferSchema", "true").load("../datasets/kddcup99/kddcup.data").drop("_c1", "_c2", "_c3", "_c41")
    val assembler = new VectorAssembler().setInputCols(frame.columns).setOutputCol("features")
    assembler.transform(frame)
  }
}