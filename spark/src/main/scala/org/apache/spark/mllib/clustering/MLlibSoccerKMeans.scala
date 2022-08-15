package org.apache.spark.mllib.clustering

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.SoccerFormulae.{DELTA_DEFAULT, alpha_formula, kplus_formula, max_subset_size_formula}
import org.apache.spark.mllib.clustering.SoccerBlackboxes.A_final
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
@Since("0.8.0")
class MLlibSoccerKMeans private(
                                 private var k: Int,
                                 private var m: Int,
                                 private var delta: Double,
                                 private var maxIterations: Int,
                                 private var epsilon: Double,
                                 private var seed: Long,
                                 private var distanceMeasure: String) extends Serializable with Logging {

  @Since("0.8.0")
  private def this(k: Int, m: Int, delta: Double, maxIterations: Int, epsilon: Double, seed: Long) =
    this(k, m, delta, maxIterations, epsilon, seed, DistanceMeasure.EUCLIDEAN)

  /**
   * Constructs a SoccerKMeans instance with default parameters: {k: 2, maxIterations: 20,
   * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random,
   * distanceMeasure: "euclidean"}.
   */
  @Since("0.8.0")
  def this() = this(2, 10, DELTA_DEFAULT, 20, 1e-4, Utils.random.nextLong(), DistanceMeasure.EUCLIDEAN)

  /**
   * Number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   *       be returned, for example, if there are fewer than k distinct points to cluster.
   */
  @Since("1.4.0")
  def getK: Int = k

  /**
   * Set the number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   *       be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
   */
  @Since("0.8.0")
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got $k")
    this.k = k
    this
  }

  /**
   * Number of machines/workers available.
   */
  @Since("1.4.0")
  def getM: Int = m

  /**
   * Set the number of machines/workers available.
   */
  @Since("0.8.0")
  def setM(m: Int): this.type = {
    require(m > 0,
      s"Number of machines must be positive but got $m")
    this.m = m
    this
  }

  /**
   * Maximum number of iterations allowed.
   */
  @Since("1.4.0")
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations allowed. Default: 20.
   */
  @Since("0.8.0")
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got $maxIterations")
    this.maxIterations = maxIterations
    this
  }

  /**
   * TODO - fix this
   * The distance threshold within which we've consider centers to have converged.
   */
  @Since("1.4.0")
  def getEpsilon: Double = epsilon

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  @Since("0.8.0")
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got $epsilon")
    this.epsilon = epsilon
    this
  }

  /**
   * TODO
   */
  @Since("1.4.0")
  def getDelta: Double = delta

  /**
   * TODO
   */
  @Since("0.8.0")
  def setDelta(delta: Double): this.type = {
    require(delta >= 0,
      s"Distance threshold must be nonnegative but got $epsilon") // TODO
    this.delta = delta
    this
  }

  /**
   * The random seed for cluster initialization.
   */
  @Since("1.4.0")
  def getSeed: Long = seed

  /**
   * Set the random seed for cluster initialization.
   */
  @Since("1.4.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * The distance suite used by the algorithm.
   */
  @Since("2.4.0")
  def getDistanceMeasure: String = distanceMeasure

  /**
   * Set the distance suite used by the algorithm.
   */
  @Since("2.4.0")
  def setDistanceMeasure(distanceMeasure: String): this.type = {
    DistanceMeasure.validateDistanceMeasure(distanceMeasure)
    this.distanceMeasure = distanceMeasure
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
  @Since("1.4.0")
  def setInitialModel(model: KMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  @Since("0.8.0")
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
    val kp = kplus_formula(k, delta, epsilon)
    var remaining_elements_count = len_N
    var alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
    val max_subset_size = max_subset_size_formula(len_N, k, epsilon, delta)
    logInfo(f"max_subset_size:$max_subset_size")

    val distanceMeasureInstance = DistanceMeasure.decodeFromString(this.distanceMeasure)
    //

    var cost = 0.0
    var iteration = 0
    val sc = data.sparkContext
    var centers = sc.emptyRDD[VectorWithNorm]

    val splits = data.randomSplit(Array.fill(m)(1.0 / m), seed)


    // TODO - consider using pyspark
    // TODO - persist RDDs between interations. This'll force spark to ""eagerly"" calculate the iterations
    while (iteration < maxIterations && remaining_elements_count > max_subset_size) {
      val bcCenters = sc.broadcast(centers.collect())
      val costAccum = sc.doubleAccumulator

      val (p1: RDD[VectorWithNorm], p2: RDD[VectorWithNorm]) = sample_P1_P2(splits, alpha)

      val (v: Double, cTmp: RDD[VectorWithNorm]) = iterate(p1, p2, alpha)

      centers = centers.union(cTmp)


      remaining_elements_count = remove_handled_points_and_return_remaining(splits, cTmp, v)

      if (remaining_elements_count == 0) {
        log.info("remaining_elements_count == 0!!")
        break
      }


      alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)


      bcCenters.destroy()

      cost = costAccum.value
      iteration += 1
    }

    val cTmp = last_iteration(splits)
    centers = centers.union(cTmp)

    val C_weights: Array[Double] = calculate_center_weights(centers, splits)
    val C_final = A_final(centers, k, C_weights)

    logInfo(s"The cost is $cost.")

    new KMeansModel(C_final.take(Integer.MAX_VALUE).map(_.vector), distanceMeasure, cost, iteration)
  }

  private def sample_P1_P2(splits: Array[RDD[VectorWithNorm]], alpha: Double): (RDD[VectorWithNorm], RDD[VectorWithNorm]) = {
    // TODO - run these two ops together
    val p1 = splits.map(s=> {
      val samp = s.sample(withReplacement = false, alpha, seed)
      // TODO - clean up logging
      log.info(f"p1 sample size: ${samp.count()}.")
      samp
    }).reduce((r1, r2) => r1.union(r2))
    val p2 = splits.map(s=>{
      val samp = s.sample(withReplacement = false, alpha, seed)
      log.info(f"p2 sample size: ${samp.count()}.")
      samp
    }).reduce((r1, r2) => r1.union(r2))
    (p1, p2)
  }

  def iterate(p1: RDD[VectorWithNorm], p2: RDD[VectorWithNorm], alpha: Double): (Double, RDD[VectorWithNorm]) = {
    (1.0, p1.sample(withReplacement = false, 1.0, seed))
  }

  def remove_handled_points_and_return_remaining(splits: Array[RDD[VectorWithNorm]], cTmp: RDD[VectorWithNorm], v: Double): Long = {
    1
  }

  def last_iteration(splits: Array[RDD[VectorWithNorm]]): RDD[VectorWithNorm] = {
    splits(0).sample(withReplacement = false, 1.0, seed)
  }

  def calculate_center_weights(centers: RDD[VectorWithNorm], splits: Array[RDD[VectorWithNorm]]): Array[Double] = {
    val len_b = splits.map(s => s.count()).sum.toInt
    var b = Array.ofDim[Double](len_b)
    b = b.map(_ => 0.1)
    b
  }
}


/**
 * Top-level methods for calling K-means clustering.
 */
@Since("0.8.0")
object MLlibSoccerKMeans {

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data               Training points as an `RDD` of `Vector` types.
   * @param k                  Number of clusters to create.
   * @param maxIterations      Maximum number of iterations allowed.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   * @param seed               Random seed for cluster initialization. Default is to generate seed based
   *                           on system time.
   */
  @Since("2.1.0")
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             initializationMode: String,
             seed: Long): KMeansModel = {
    new MLlibSoccerKMeans()
      .setK(k)
      .setDelta(k)
      .setEpsilon(k)
      .setMaxIterations(maxIterations)
      .setSeed(seed)
      .run(data)
  }

  /**
   * Trains a soccer k-means model using the given set of parameters.
   *
   * @param data               Training points as an `RDD` of `Vector` types.
   * @param k                  Number of clusters to create.
   * @param maxIterations      Maximum number of iterations allowed.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   */
  @Since("2.1.0")
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             initializationMode: String): KMeansModel = {
    new MLlibSoccerKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .run(data)
  }

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  @Since("0.8.0")
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int): KMeansModel = {
    new MLlibSoccerKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .run(data)
  }

  private[spark] def validateInitMode(initMode: String): Boolean = {
    initMode match {
      case KMeans.RANDOM => true
      case KMeans.K_MEANS_PARALLEL => true
      case _ => false
    }
  }
}
