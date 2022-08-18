package org.apache.spark.mllib.clustering

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering.SoccerFormulae._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

import scala.util.control.Breaks.break

/**
 * K-means clustering with a k-means++ like initialization mode
 * (the k-means|| algorithm by Bahmani et al).
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


  private def this(k: Int, m: Int, delta: Double, maxIterations: Int, epsilon: Double, seed: Long) =
    this(k, m, delta, maxIterations, epsilon, seed, DistanceMeasure.EUCLIDEAN)

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

  private var distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString(distanceMeasure)

  /**
   * Set the distance suite used by the algorithm.
   */
  def setDistanceMeasure(distanceMeasure: String): this.type = {
    DistanceMeasure.validateDistanceMeasure(distanceMeasure)
    this.distanceMeasure = distanceMeasure
    this.distanceMeasureInstance = DistanceMeasure.decodeFromString(this.distanceMeasure)
    this
  }

  // Initial cluster centers can be provided as a KMeansModel object rather than using the
  // random or k-means|| initializationMode
  private var initialModel: Option[KMeansModel] = None

  /**
   * Set the initial starting point, bypassing the random initialization or k-means||
   * The condition model.k == this.k must be met, failure results
   * in an IllegalArgumentException.
   */
  def setInitialModel(model: KMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
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
    val kplus = kplus_formula(k, delta, epsilon) // TODO - calc this in ctor
    var remaining_elements_count = len_N
    var alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
    val max_subset_size = max_subset_size_formula(len_N, k, epsilon, delta)
    logInfo(f"max_subset_size:$max_subset_size")

    var iteration = 0
    val sc = data.sparkContext
    var centers = sc.emptyRDD[VectorWithNorm]

    // TODO - automatically detect the best number of splits
    val splits = data.randomSplit(Array.fill(m)(1.0 / m), seed)
    var unhandled_data_splits = splits


    // TODO - persist RDDs between interations. This'll force spark to ""eagerly"" calculate the iterations
    while (iteration < maxIterations && remaining_elements_count > max_subset_size) {


      val (p1, p2) = sample_P1_P2(unhandled_data_splits, alpha)

      val (v, cTmp) = EstProc(p1, p2, alpha, k, kplus)


      unhandled_data_splits = unhandled_data_splits.map(s => removeHandled(s, cTmp, v))
      remaining_elements_count = unhandled_data_splits.map(s => s.count()).sum

      unhandled_data_splits.foreach(s => logInfo(f"Iter $iteration: split has remaining ${s.count()} elems"))
      logInfo(f"Total remaining: $remaining_elements_count")
      if (remaining_elements_count == 0) {
        log.info("remaining_elements_count == 0!!")
        break
      }

      alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
      centers ++= cTmp
      iteration += 1
    }

    val cTmp = last_iteration(unhandled_data_splits)
    centers ++= cTmp

    val C_weights = calculate_center_weights(centers, splits)
    val C_final = A_final(centers, k, C_weights)

    val trainingCost = 1000.0 // TODO
    new KMeansModel(C_final.map(_.vector), distanceMeasure, trainingCost, iteration)
  }

  private def sample_P1_P2(splits: Array[RDD[VectorWithNorm]], alpha: Double): (RDD[VectorWithNorm], RDD[VectorWithNorm]) = {
    // TODO - run these two ops together
    // Maybe take |p1+p2| elems and then shuffle them here on the coordinator
    val p1 = splits
      .map(s => s.sample(withReplacement = false, alpha, seed))
      .reduce((r1, r2) => r1.union(r2))

    val p2 = splits
      .map(s => s.sample(withReplacement = false, alpha, seed))
      .reduce((r1, r2) => r1.union(r2))

    (p1, p2)
  }


  /**
   * calculates a rough clustering on P1. Estimates the risk of the clusters on P2.
   * Emits the cluster and the ~risk.
   */
  private def EstProc(p1: RDD[VectorWithNorm], p2: RDD[VectorWithNorm], alpha: Double, k: Int, kp: Int): (Double, RDD[VectorWithNorm]) = {
    val cTmp = A_inner(p1, kp)

    val phi_alpha = phi_alpha_formula(alpha, k, delta, epsilon)
    val r = r_formula(alpha, k, phi_alpha)
    val Rr = risk_truncated(p2, cTmp, r)

    psi = Math.max((2 / (3 * alpha)) * Rr, psi)
    val v = v_formula(psi, k, phi_alpha)
    if (v == 0.0) log.error("Bad! v == 0.0")
    (v, cTmp)
  }

  private def A_inner(n: RDD[VectorWithNorm], k: Int): RDD[VectorWithNorm] = {
    val algo = new KMeans()
      .setK(k)
      .setSeed(seed)

    // TODO - optimize?
    val sc = n.context
    log.info("================================= starting A_inner =================================")
    val inner_centers = algo.run(n.map(v => v.vector)).clusterCenters.map(v => new VectorWithNorm(v, Vectors.norm(v, 2.0)))
    log.info("=================================    ended A_inner =================================")
    sc.parallelize(inner_centers)
  }

  private def A_final(centers: RDD[VectorWithNorm], k: Int, center_weights: RDD[Double]): Array[VectorWithNorm] = {
    val algo = new KMeans()
      .setK(k)
      .setSeed(seed)

    log.info("================================= starting A_final =================================")
    val weighted_centers = centers.map(c => c.vector).zip(center_weights)
    val final_centers = algo.runWithWeight(weighted_centers, handlePersistence = false, Option.empty).clusterCenters.map(v => new VectorWithNorm(v, Vectors.norm(v, 2.0)))
    log.info("================================= finished A_final =================================")
    final_centers
  }


  private def risk_truncated(p2: RDD[VectorWithNorm], C: RDD[VectorWithNorm], r: Int): Double = {
    if (r >= p2.count()) {
      return 0 // The "trivial risk"
    }

    val distances = pairwise_distances_argmin_min_squared(p2, C)
    distances.takeOrdered(distances.count().toInt - r).sum
  }


  private def removeHandled(s: RDD[VectorWithNorm], cTmp: RDD[VectorWithNorm], v: Double): RDD[VectorWithNorm] = {
    val centers = cTmp.collect()
    s.filter(p => distanceMeasureInstance.pointCost(centers, p) < v)
  }

  private def last_iteration(splits: Array[RDD[VectorWithNorm]]): RDD[VectorWithNorm] = {
    logInfo("Starting last iteration")
    val remaining_data = splits.reduce((s1, s2) => s1.union(s2))
    val cTmp =
      if (remaining_data.count() <= k) // TODO - support this - or (self._blackbox == "ScalableKMeans" andlen(N_remaining) <= l)):
        remaining_data
      else
        A_inner(remaining_data, k /* TODO - should this be kplus?*/)
    logInfo("Finished last iteration")
    cTmp
  }

  private def calculate_center_weights(centers: RDD[VectorWithNorm], splits: Array[RDD[VectorWithNorm]]): RDD[Double] = {
    val collected_centers = centers.collect()
    val final_center_counts = splits.map(s => {
      val center_counts_map = collection.mutable.Map[Int, Double]().withDefaultValue(0.0)
      s.foreach(p => {
        val (bestCenter, _) = distanceMeasureInstance.findClosest(collected_centers, p)
        center_counts_map(bestCenter) = center_counts_map(bestCenter) + 1
      })
      center_counts_map
    }).reduce((m1, m2) => reduceCountsMap(m1, m2))
    val sc = centers.context
    sc.parallelize(final_center_counts.toSeq.sorted.map(_._2))
  }

  def reduceCountsMap(m1: collection.mutable.Map[Int, Double], m2: collection.mutable.Map[Int, Double]): collection.mutable.Map[Int, Double] = {
    val merged = m1.toSeq ++ m2.toSeq
    val grouped = merged.groupBy(_._1)
    val recounted = collection.mutable.Map(grouped.view.mapValues(_.map(_._2).sum).toSeq: _*)
    recounted
  }


  private def pairwise_distances_argmin_min_squared(X: RDD[VectorWithNorm], Y: RDD[VectorWithNorm]): RDD[Double] = { // TODO - consider using Arrays here to be explicitly local
    val ys = Y.collect()
    X.map { point =>
      val (_, cost) = distanceMeasureInstance.findClosest(ys, point)
      Math.pow(cost, 2)
    }
  }
}
