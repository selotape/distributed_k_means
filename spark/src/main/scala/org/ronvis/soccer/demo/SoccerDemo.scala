package org.ronvis.soccer.demo

import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel, SoccerKMeans}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, SparkSession}

object SoccerDemo {
  lazy val log: Logger = LogManager.getLogger(this.getClass)


  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master(master = "local[*]")
      .appName(name = "SOCCER example")
      .getOrCreate()

    //    val dataset = loadSampleKMeansDataset(spark)
    //    val dataset = loadCsvDataset(spark, "../datasets/kddcup99/kddcup.data", limit = 10000)
    //    val dataset = loadCsvDataset(spark, csvPath = "../datasets/higgs/HIGGS_top20k.csv")
    val dataset = loadCsvDataset(spark, csvPath = "../datasets/higgs/HIGGS.csv", limit = 100000)
    val seed = 1L
    val k = 25

    for (i <- 1 to 3) {
      log.info(f"================== STARTING SOCCER KMEANS run $i ==================")
      val soccerKmeans = new SoccerKMeans()
        .setK(k)
        .setM(4)
        .setTol(0.05) // aka - epsilon
        .setDelta(0.1)
        .setSeed(seed)
        .setMaxIter(3)
      fitAndEvaluate(soccerKmeans, dataset)
      log.info(f"================== FINISHED SOCCER KMEANS run $i ==================")
    }
    log.info("============================================================")
    log.info("============================================================")
    log.info("============================================================")
    log.info("============================================================")
    log.info("============================================================")

    for (i <- 1 to 3) {
      log.info(f"================== STARTING LEGACY KMEANS run $i ==================")
      val boringOldschoolKmeans = new KMeans().setK(k).setSeed(seed)
      fitAndEvaluate(boringOldschoolKmeans, dataset)
      log.info(f"================== FINISHED LEGACY KMEANS run $i ==================")
    }


  }

  def fitAndEvaluate(kmeans: Estimator[_ <: Model[_]], dataset: DataFrame): Unit = {
    val startTimeMillis = System.currentTimeMillis()
    val model = kmeans.fit(dataset).asInstanceOf[KMeansModel]
    val totalSecs = (System.currentTimeMillis() - startTimeMillis) / 1000.0
    val C = model.clusterCenters
    val risk_final = model.summary.trainingCost
    val iterations = model.summary.numIter
    log.info(f"Reached trainingCost ${model.summary.trainingCost} after ${model.summary.numIter} iterations and $totalSecs elapsed wallclock seconds")
    log.info("test_name,k,dt,m,ep,l,len(C),iterations,risk,risk_final,reducers_time,total_time")
    log.info(f"test_name,k,dt,m,ep,l,${C.length},$iterations,risk,$risk_final,reducers_time,$totalSecs")
  }

  def loadSampleKMeansDataset(spark: SparkSession): DataFrame = {
    spark.read.format("libsvm").load("../datasets/sample_kmeans_data.txt")
  }

  def loadCsvDataset(spark: SparkSession, csvPath: String, limit: Int = Int.MaxValue): DataFrame = {
    val frame = spark.read.format(source = "csv").option("inferSchema", "true").option("mode", "DROPMALFORMED").load(csvPath).limit(limit)
    val numericFrame = retainNumericColsDropNaAndLog(frame)
    val assembler = new VectorAssembler().setInputCols(numericFrame.columns).setOutputCol("features")
    assembler.transform(numericFrame)
  }


  private def retainNumericColsDropNaAndLog(df: DataFrame): DataFrame = {
    logDf("Before cleanup: ", df)
    val numericDf = removeNonNumericColumns(df)
    val dfWithoutNa = numericDf.na.drop()
    logDf(" After cleanup: ", dfWithoutNa)
    dfWithoutNa
  }

  private val NON_NUMERIC_COLUM_TYPES = Set("string", "boolean")

  private def removeNonNumericColumns(df: DataFrame): DataFrame = {
    val numericColumns = df.columns.filter(cName => !NON_NUMERIC_COLUM_TYPES.contains(df.schema(cName).dataType.typeName))
    df.select(numericColumns(0), numericColumns: _*)
  }

  private def logDf(prefix: String, df: DataFrame): Unit = {
    log.info(prefix + f"count:${df.count()} " + df.columns.map(c => df.schema(c).dataType.typeName).mkString("Array(", ", ", ")"))
  }

}