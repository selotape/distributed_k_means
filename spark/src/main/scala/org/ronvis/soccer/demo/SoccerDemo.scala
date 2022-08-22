package org.ronvis.soccer.demo

import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.ml.clustering.{KMeans, SoccerKMeans}
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

    //    val dataset = getSampleKMeansDataset(spark)
    val dataset = getKddCup99Dataset(spark)
    val seed = 1L
    val k = 25

    log.info("================== STARTING SOCCER KMEANS ==================")
    val soccerKmeans = new SoccerKMeans()
      .setK(k)
      .setM(50)
      .setTol(0.05) // aka - epsilon
      .setDelta(0.1)
      .setSeed(seed)
      .setMaxIter(3)
    //    fitAndEvaluate(soccerKmeans, dataset)
    log.info("================== FINISHED SOCCER KMEANS ==================")

    log.info("================== STARTING LEGACY KMEANS ==================")
    val boringOldschoolKmeans = new KMeans().setK(k).setSeed(seed)
    //    fitAndEvaluate(boringOldschoolKmeans, dataset)
    log.info("================== FINISHED LEGACY KMEANS ==================")
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
    val frame = spark.read.format("csv").option("inferSchema", "true").load("../datasets/kddcup99/kddcup.data")
    val numericFrame = retainNumericColumnsAndLog(frame)
    val assembler = new VectorAssembler().setInputCols(numericFrame.columns).setOutputCol("features")
    assembler.transform(numericFrame)
  }

  def getKddCup99DatasetTop10k(spark: SparkSession): DataFrame = {
    getKddCup99Dataset(spark).limit(10000)
  }

  private def retainNumericColumnsAndLog(df: DataFrame): DataFrame = {
    printColumnTypes("Before cleanup: ", df)
    val cleanedDf = removeNonNumericColumns(df)
    printColumnTypes(" After cleanup: ", cleanedDf)
    cleanedDf
  }

  private val NON_NUMERIC_COLUM_TYPES = Set("string", "boolean")

  private def removeNonNumericColumns(df: DataFrame): DataFrame = {
    val numericColumns = df.columns.filter(cName => !NON_NUMERIC_COLUM_TYPES.contains(df.schema(cName).dataType.typeName))
    df.select(numericColumns(0), numericColumns: _*)
  }

  private def printColumnTypes(prefix: String, df: DataFrame): Unit = {
    log.info(prefix + df.columns.map(c => df.schema(c).dataType.typeName).mkString("Array(", ", ", ")"))
  }

}