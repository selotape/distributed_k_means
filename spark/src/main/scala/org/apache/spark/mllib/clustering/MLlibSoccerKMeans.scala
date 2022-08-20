package org.apache.spark.mllib.clustering

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering.SoccerFormulae._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

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
    var alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
    val max_subset_size = max_subset_size_formula(len_N, k, epsilon, delta)
    require(max_subset_size > 0, s"max_subset_size must be nonnegative but got $max_subset_size")
    logInfo(f"max_subset_size:$max_subset_size")
    var iteration = 0
    val sc = data.sparkContext
    var centers = sc.emptyRDD[VectorWithNorm]
    val splits = data.randomSplit(Array.fill(m)(1.0 / m), seed1).par // TODO - automatically detect the best number of splits instead of relying on m
    var unhandled_data_splits = splits


    // TODO - persist RDDs between interations. This'll force spark to ""eagerly"" calculate the iterations
    while (iteration < maxIterations && remaining_elements_count > max_subset_size) {

      val (p1, p2) = sample_P1_P2(unhandled_data_splits, alpha)
      val (v, cTmp) = EstProc(p1, p2, alpha)

      unhandled_data_splits.foreach(s => logInfo(f"Iter $iteration: split has remaining ${s.count()} elems"))
      unhandled_data_splits = unhandled_data_splits.map(s => removeHandled(s, cTmp, v))
      unhandled_data_splits.foreach(s => logInfo(f"Iter $iteration: split has remaining ${s.count()} elems"))

      remaining_elements_count = unhandled_data_splits.map(s => s.count()).sum
      logInfo(f"remaining_elements_count: $remaining_elements_count")
      centers ++= cTmp
      if (remaining_elements_count != 0) {
        alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
        logInfo(f"At end of iter $iteration there are ${centers.count()} centers")
        iteration += 1
      }
    }

    val cTmp = last_iteration(unhandled_data_splits)
    centers ++= cTmp

    val C_weights = calculate_center_weights(centers, splits)
    val C_final = A_final(centers, C_weights)

    val trainingCost = 1000.0 // TODO
    new KMeansModel(C_final.map(_.vector), distanceMeasure, trainingCost, iteration)
  }

  private def sample_P1_P2(splits: ParArray[RDD[VectorWithNorm]], alpha: Double): (RDD[VectorWithNorm], RDD[VectorWithNorm]) = {
    // TODO - run these two ops together. Maybe take |p1+p2| elems and then shuffle them here on the coordinator
    val p1p2 = splits
      .map(s => s.sample(withReplacement = false, 2 * alpha, seed1))
      .map(p12 => p12.randomSplit(Array(1, 1)))
      .reduce((left, right) => Array(left(0).union(right(0)), left(1).union(right(1))))

    (p1p2(0), p1p2(1))
  }


  /**
   * calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
   * Emits the cluster and the ~risk.
   */
  private def EstProc(p1: RDD[VectorWithNorm], p2: RDD[VectorWithNorm], alpha: Double) = {
    val cTmp = A_inner(p1)

    val phi_alpha = phi_alpha_formula(alpha, k, delta, epsilon)
    val r = r_formula(alpha, k, phi_alpha)
    val Rr = risk_truncated(p2, cTmp, r)

    psi = Math.max((2 / (3 * alpha)) * Rr, psi)
    val v = v_formula(psi, k, phi_alpha)
    if (v == 0.0) log.error("Bad! v == 0.0")
    (v, cTmp)
  }

  private def A_inner(n: RDD[VectorWithNorm]): RDD[VectorWithNorm] = {
    log.info("================================= starting A_inner =================================")
    val algo = createInnerKMeans(kplus)
    val inner_centers = algo.run(n.map(v => v.vector)).clusterCenters.map(v => new VectorWithNorm(v, Vectors.norm(v, 2.0))) // TODO - optimize multiple mappings and object creations?
    log.info(f"================================= ended A_inner with ${inner_centers.length} centers =================================")
    n.context.parallelize(inner_centers)
  }

  private def A_final(centers: RDD[VectorWithNorm], center_weights: RDD[Double]) = {
    log.info(s"================================= starting A_final with ${centers.count()} centers and ${center_weights.count()} weights =================================") // TODO - clean up loglines
    val algo = createInnerKMeans(this.k)
    val weighted_centers = centers.repartition(1).map(c => c.vector).zip(center_weights.repartition(1))
    val final_centers = algo.runWithWeight(weighted_centers, handlePersistence = false, Option.empty).clusterCenters.map(v => new VectorWithNorm(v, Vectors.norm(v, 2.0)))
    log.info(f"================================= finished A_final with ${final_centers.length} centers =================================")
    final_centers
  }

  private def createInnerKMeans(innerK: Int): KMeans = {
    new KMeans()
      .setK(innerK)
      .setSeed(seed2)
      .setInitializationMode(KMEANS_INIT_MODE)

  }


  private def risk_truncated(p2: RDD[VectorWithNorm], C: RDD[VectorWithNorm], r: Int): Double = {
    if (r >= p2.count()) {
      return 0 // The "trivial risk"
    }

    val distances = pairwise_distances_argmin_min_squared(p2, C)
    distances.takeOrdered(distances.count().toInt - r).sum
  }


  private def pairwise_distances_argmin_min_squared(X: RDD[VectorWithNorm], Y: RDD[VectorWithNorm]): RDD[Double] = { // TODO - consider using Arrays here to be explicitly local
    val ys = Y.collect()
    X.map { point =>
      val (_, cost) = distanceMeasureInstance.findClosest(ys, point)
      Math.pow(cost, 2)
    }
  }

  private def removeHandled(s: RDD[VectorWithNorm], cTmp: RDD[VectorWithNorm], v: Double): RDD[VectorWithNorm] = {
    val centers = cTmp.collect()
    s.filter(p => distanceMeasureInstance.pointCost(centers, p) > v)
  }

  private def last_iteration(splits: ParArray[RDD[VectorWithNorm]]): RDD[VectorWithNorm] = {
    logInfo("Starting last iteration")
    val remaining_data = splits.reduce((s1, s2) => s1.union(s2))
    val cTmp =
      if (remaining_data.count() <= k) // TODO - support this - or (self._blackbox == "ScalableKMeans" and len(N_remaining) <= l)):
        remaining_data
      else
        A_inner(remaining_data) // TODO - should this use kplus?
    logInfo("Finished last iteration")
    cTmp
  }

  private def calculate_center_weights(centers: RDD[VectorWithNorm], splits: ParArray[RDD[VectorWithNorm]]): RDD[Double] = {
    val collected_centers = centers.collect()
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
    sc.parallelize(weights)
  }


  private def reduceCountsMap(m1: scala.collection.Map[Int, Long], m2: scala.collection.Map[Int, Long]): scala.collection.Map[Int, Long] = {
    val merged = m1.toSeq ++ m2.toSeq
    val grouped = merged.groupBy(_._1)
    val recounted = scala.collection.Map(grouped.view.mapValues(_.map(_._2).sum).toSeq: _*)
    recounted
  }
}
