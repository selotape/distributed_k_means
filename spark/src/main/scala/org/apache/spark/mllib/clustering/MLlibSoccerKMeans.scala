package org.apache.spark.mllib.clustering

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering.SoccerFormulae._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.util.{DoubleAccumulator, Utils}

import scala.collection.parallel.CollectionConverters._
import scala.collection.parallel.mutable.ParArray

/**
 * TODO - write this doc
 *
 * This is an iterative algorithm that will make multiple passes over the data, so any RDDs given
 * to it should be cached by the user.
 */
class MLlibSoccerKMeans private(
                                 private var k: Int,
                                 private var m: Int,
                                 private var delta: Double,
                                 private var maxIterations: Int,
                                 private var epsilon: Double,
                                 private var seed: Long,
                                 private var distanceMeasure: String,
                                 private var psi: Double = 0) extends Serializable with Logging {

  private val xorShiftRandom = new XORShiftRandom(this.seed)
  private val seed1: Long = xorShiftRandom.nextInt() // TODO - remove this.seed and keep only seed1 & 2
  private val seed2: Long = xorShiftRandom.nextInt()
  private val kplus: Int = kplus_formula(k, delta, epsilon)
  private var distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString(distanceMeasure)

  /**
   * Constructs a SoccerKMeans instance with default parameters: {k: 2, maxIterations: 20,
   * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random,
   * distanceMeasure: "euclidean"}.
   */
  def this() = this(2, 10, DELTA_DEFAULT, 20, 1e-4, Utils.random.nextLong(), DistanceMeasure.EUCLIDEAN)

  /**
   * Number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   *       be returned, for example, if there are fewer than k distinct points to cluster.
   */
  def getK: Int = k

  /**
   * Set the number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   *       be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
   */
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got $k")
    this.k = k
    this
  }

  /**
   * Number of machines/workers available.
   */
  def getM: Int = m

  /**
   * TODO - make this autodiscorevable
   * Set the number of machines/workers available.
   */
  def setM(m: Int): this.type = {
    require(m > 0,
      s"Number of machines must be positive but got $m")
    this.m = m
    this
  }

  /**
   * Maximum number of iterations allowed.
   */
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations allowed. Default: 20.
   */
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got $maxIterations")
    this.maxIterations = maxIterations
    this
  }

  /**
   * TODO
   */
  def getEpsilon: Double = epsilon

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got $epsilon")
    this.epsilon = epsilon
    this
  }

  /**
   * TODO
   */
  def getDelta: Double = delta

  /**
   * TODO
   */
  def setDelta(delta: Double): this.type = {
    require(delta >= 0,
      s"Distance threshold must be nonnegative but got $epsilon") // TODO
    this.delta = delta
    this
  }

  /**
   * The random seed for cluster initialization.
   */
  def getSeed: Long = seed

  /**
   * Set the random seed for cluster initialization.
   */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * The distance suite used by the algorithm.
   */
  def getDistanceMeasure: String = distanceMeasure

  /**
   * Set the distance suite used by the algorithm.
   */
  def setDistanceMeasure(distanceMeasure: String): this.type = {
    DistanceMeasure.validateDistanceMeasure(distanceMeasure)
    this.distanceMeasure = distanceMeasure
    this.distanceMeasureInstance = DistanceMeasure.decodeFromString(this.distanceMeasure)
    this
  }


  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  def run(data: RDD[Vector]): KMeansModel = {
    val instances = data.map(point => (point, 1.0))
    runWithWeight(instances)
  }

  private[spark] def runWithWeight(
                                    instances: RDD[(Vector, Double)]): KMeansModel = {
    val norms = instances.map { case (v, _) => Vectors.norm(v, 2.0) }
    val vectors = instances.zip(norms)
      .map { case ((v, w), norm) => new VectorWithNorm(v, norm, w) }
    val model = runAlgorithmWithWeight(vectors)

    model
  }

  /**
   * Implementation of SOCCER K-Means algorithm.
   */
  private def runAlgorithmWithWeight(data: RDD[VectorWithNorm]): KMeansModel = {

    val len_N = data.count()
    var remaining_elements_count = len_N
    var alpha: Double = -1.0
    val max_subset_size = max_subset_size_formula(len_N, k, epsilon, delta)
    require(max_subset_size > 0, s"max_subset_size must be nonnegative but got $max_subset_size")
    logInfo(f"max_subset_size:$max_subset_size")
    var iteration = 0
    val sc = data.sparkContext
    var centers = sc.emptyRDD[VectorWithNorm]
    val splits = data.randomSplit(Array.fill(m)(1.0 / m), seed1).par // TODO - automatically detect the best number of splits instead of relying on m
    var unhandled_data_splits = splits
    val risk = sc.doubleAccumulator("Nonfinal Risk")


    // TODO - consider persisting RDDs between interations. This'll force spark to ""eagerly"" calculate the iterations
    while (iteration < maxIterations && remaining_elements_count > max_subset_size) {
      alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
      val (p1, p2) = sample_P1_P2(unhandled_data_splits, alpha)
      val (v, cTmp) = EstProc(p1, p2, alpha)

      val removeHandledStartTimeMillis = System.currentTimeMillis()
      unhandled_data_splits = unhandled_data_splits.map(s => removeHandledAndCountHandledRisk(s, cTmp, v, risk))
      remaining_elements_count = unhandled_data_splits.map(s => s.count()).sum
      log.info(f"================================= removeHandledAndCountHandledRisk in iter $iteration took ${elapsedSecs(removeHandledStartTimeMillis)} seconds")

      logInfo(f"iter $iteration: cTmp.count=${cTmp.count()}. remaining_elements_count=$remaining_elements_count. alpha=$alpha. p1.count=${p1.count()}. v=$v")
      centers ++= cTmp
      iteration += 1
    }

    if (remaining_elements_count > 0) {
      val cTmp = last_iteration(unhandled_data_splits, risk)
      centers ++= cTmp
    }
    logInfo(f"Non-final risk is ${risk.value}")

    val C_weights = calculate_center_weights(centers, splits)
    val C_final = A_final(centers, C_weights)
    val trainingCost = measureTrainingCost(C_final, data)

    val newKmeansModelStartTimeMillis = System.currentTimeMillis()
    val kmeansModel = new KMeansModel(C_final.map(_.vector), distanceMeasure, trainingCost, iteration)
    log.info(f"================================= new KMeansModel() took ${elapsedSecs(newKmeansModelStartTimeMillis)} seconds")
    kmeansModel
  }

  private def sample_P1_P2(splits: ParArray[RDD[VectorWithNorm]], alpha: Double): (RDD[VectorWithNorm], RDD[VectorWithNorm]) = {
    val startTimeMillis = System.currentTimeMillis()
    val p1p2 = splits
      .map(s => s.sample(withReplacement = false, 2 * alpha, seed1))
      .map(p12 => p12.randomSplit(Array(1, 1)))
      .reduce((left, right) => Array(left(0).union(right(0)), left(1).union(right(1))))

    log.info(f"================================= sample_P1_P2 took ${elapsedSecs(startTimeMillis)} seconds")
    (p1p2(0), p1p2(1))
  }

  /**
   * calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
   * Emits the cluster and the ~risk.
   */
  private def EstProc(p1: RDD[VectorWithNorm], p2: RDD[VectorWithNorm], alpha: Double): (Double, RDD[VectorWithNorm]) = {
    val startTimeMillis = System.currentTimeMillis()
    val cTmp = A_inner(p1)

    val phi_alpha = phi_alpha_formula(alpha, k, delta, epsilon)
    val r = r_formula(alpha, k, phi_alpha)
    val rTruncated = risk_truncated(p2, cTmp, r)

    psi = Math.max((2 / (3 * alpha)) * rTruncated, psi)
    val v = v_formula(psi, k, phi_alpha)
    if (v == 0.0) log.error("Bad! v == 0.0")
    log.info(f"================================= EstProc took ${elapsedSecs(startTimeMillis)} seconds")
    (v, cTmp)
  }


  private def A_inner(n: RDD[VectorWithNorm]): RDD[VectorWithNorm] = {
    val startTimeMillis = System.currentTimeMillis()
    log.info(s"================================= starting A_inner on ${n.count()} elements")
    val algo = createInnerKMeans(kplus)
    // TODO - optimize multiple mappings and object creations?
    // TODO - make sure kmeans runs only once
    val inner_centers = algo.run(n.map(v => v.vector)).clusterCenters.map(v => new VectorWithNorm(v, Vectors.norm(v, 2.0)))
    log.info(f"================================= ended A_inner with ${inner_centers.length} centers and took ${elapsedSecs(startTimeMillis)} seconds")
    n.context.parallelize(inner_centers)
  }

  private def A_final(centers: RDD[VectorWithNorm], center_weights: RDD[Double]) = {
    log.info(s"================================= starting A_final with ${centers.count()} centers and ${center_weights.count()} weights") // TODO - clean up loglines
    val startTimeMillis = System.currentTimeMillis()
    val algo = createInnerKMeans(this.k)
    val weighted_centers = centers.repartition(1).map(c => c.vector).zip(center_weights.repartition(1))
    val final_centers = algo.runWithWeight(weighted_centers, handlePersistence = false, Option.empty).clusterCenters.map(v => new VectorWithNorm(v, Vectors.norm(v, 2.0)))
    log.info(f"================================= finished A_final with ${final_centers.length} centers which took ${elapsedSecs(startTimeMillis)} seconds")
    final_centers
  }

  private def createInnerKMeans(innerK: Int): KMeans = {
    new KMeans()
      .setK(innerK)
      .setSeed(seed2)
//      .setEpsilon(0.1) // TODO - go over with Hess
//      .setInitializationMode(KMEANS_INIT_MODE)
  }

  private def risk_truncated(p2: RDD[VectorWithNorm], C: RDD[VectorWithNorm], r: Int): Double = {
    val startTimeMillis = System.currentTimeMillis()
    if (r >= p2.count()) {
      return 0 // The "trivial risk"
    }

    val distances = pairwise_distances_argmin_min_squared(p2, C)
    val the_sum = distances.takeOrdered(distances.count().toInt - r).sum
    log.info(f"================================= risk_truncated took ${elapsedSecs(startTimeMillis)} seconds")
    the_sum
  }

  private def pairwise_distances_argmin_min_squared(X: RDD[VectorWithNorm], Y: RDD[VectorWithNorm]): RDD[Double] = { // TODO - consider using Arrays here to be explicitly local
    val ys = Y.collect()
    X.map { point =>
      val (_, cost) = distanceMeasureInstance.findClosest(ys, point)
      Math.pow(cost, 2)
    }
  }

  private def removeHandledAndCountHandledRisk(s: RDD[VectorWithNorm], cTmp: RDD[VectorWithNorm], v: Double, riskAccumulator: DoubleAccumulator): RDD[VectorWithNorm] = {
    val centers = cTmp.collect()
    s.filter(p => {
      val cost = distanceMeasureInstance.pointCost(centers, p)
      if (cost <= v) {
        riskAccumulator.add(cost) // TODO - go over this with Tom
      }
      cost > v
    })
  }

  private def last_iteration(splits: ParArray[RDD[VectorWithNorm]], risk: DoubleAccumulator /*TODO - count risk*/): RDD[VectorWithNorm] = {
    log.info("================================= Starting last iteration")
    val startTimeMillis = System.currentTimeMillis()
    val remaining_data = splits.reduce((s1, s2) => s1.union(s2))
    val cTmp =
      if (remaining_data.count() <= k) // TODO - support this - or (self._blackbox == "ScalableKMeans" and len(N_remaining) <= l)):
        remaining_data
      else
        A_inner(remaining_data) // TODO - should this use kplus?
    log.info(f"================================= Finished last iteration which took ${elapsedSecs(startTimeMillis)} seconds")
    cTmp
  }

  private def calculate_center_weights(centers: RDD[VectorWithNorm], splits: ParArray[RDD[VectorWithNorm]]): RDD[Double] = {
    val startTimeMillis = System.currentTimeMillis()
    val collected_centers = centers.collect() // TODO - broadcast collected_centers so it's sent to each worker only once and not once per task
    val final_center_counts =
      splits.map(
        s => {
          s.map(
            p => {
              val (bestCenter, _) = distanceMeasureInstance.findClosest(collected_centers, p)
              bestCenter
            })
            .countByValue()
        }).reduce(
        (m1, m2) => reduceCountsMap(m1, m2))
    val sc = centers.context
    val weights = List.range(0, collected_centers.length).map(i => final_center_counts.getOrElse(i, 0L)).map(l => l.doubleValue)
    if (weights.contains(0.0)) {
      logError("Bad! weights contain a 0.0")
    }
    val weightsRdd = sc.parallelize(weights)
    log.info(f"================================= calculate_center_weights took ${elapsedSecs(startTimeMillis)} seconds")
    weightsRdd
  }

  private def reduceCountsMap(m1: scala.collection.Map[Int, Long], m2: scala.collection.Map[Int, Long]): scala.collection.Map[Int, Long] = {
    val merged = m1.toSeq ++ m2.toSeq
    val grouped = merged.groupBy(_._1)
    val recounted = scala.collection.Map(grouped.view.mapValues(_.map(_._2).sum).toSeq: _*)
    recounted
  }


  def measureTrainingCost(C_final: Array[VectorWithNorm], data: RDD[VectorWithNorm]): Double = {
    val startTimeMillis = System.currentTimeMillis()
    val theSum = data.map(v => distanceMeasureInstance.pointCost(C_final, v)).sum()
    log.info(f"================================= measureTrainingCost took ${elapsedSecs(startTimeMillis)} seconds")
    theSum
  }

}
