package org.apache.spark.mllib.clustering

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.SoccerFormulae.{DELTA_DEFAULT, alpha_formula, kplus_formula, max_subset_size_formula}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

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
                       private var delta: Double,
                       private var maxIterations: Int,
                       private var epsilon: Double,
                       private var seed: Long,
                       private var distanceMeasure: String) extends Serializable with Logging {

  @Since("0.8.0")
  private def this(k: Int, delta: Double, maxIterations: Int, epsilon: Double, seed: Long) =
    this(k, delta, maxIterations, epsilon, seed, DistanceMeasure.EUCLIDEAN)

  /**
   * Constructs a SoccerKMeans instance with default parameters: {k: 2, maxIterations: 20,
   * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random,
   * distanceMeasure: "euclidean"}.
   */
  @Since("0.8.0")
  def this() = this(2, DELTA_DEFAULT, 20, 1e-4, Utils.random.nextLong(), DistanceMeasure.EUCLIDEAN)

  /**
   * Number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster.
   */
  @Since("1.4.0")
  def getK: Int = k

  /**
   * Set the number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
   */
  @Since("0.8.0")
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got $k")
    this.k = k
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
    val handlePersistence = data.getStorageLevel == StorageLevel.NONE
    runWithWeight(instances, handlePersistence, None)
  }

  private[spark] def runWithWeight(
                                    instances: RDD[(Vector, Double)],
                                    handlePersistence: Boolean,
                                    instr: Option[Instrumentation]): KMeansModel = {
    val norms = instances.map { case (v, _) => Vectors.norm(v, 2.0) }
    val vectors = instances.zip(norms)
      .map { case ((v, w), norm) => new VectorWithNorm(v, norm, w) }

    if (handlePersistence) {
      vectors.persist(StorageLevel.MEMORY_AND_DISK)
    } else {
      // Compute squared norms and cache them.
      norms.persist(StorageLevel.MEMORY_AND_DISK)
    }
    val model = runAlgorithmWithWeight(vectors, instr)
    if (handlePersistence) { vectors.unpersist() } else { norms.unpersist() }

    model
  }

  /**
   * Implementation of SOCCER K-Means algorithm.
   */
  private def runAlgorithmWithWeight(
                                      data: RDD[VectorWithNorm],
                                      instr: Option[Instrumentation]): KMeansModel = {

    val kp = kplus_formula(k, delta, epsilon)
    val len_N = data.count()
    val remaining_elements_count = len_N
    val alpha = alpha_formula(len_N, k, epsilon, delta, remaining_elements_count)
    val max_subset_size = max_subset_size_formula(len_N, k, epsilon, delta)
    val initStartTime = System.nanoTime()

    val distanceMeasureInstance = DistanceMeasure.decodeFromString(this.distanceMeasure)

    val centers = new Array[VectorWithNorm](0)
    val numFeatures = data.first().vector.size

    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization took $initTimeInSeconds%.3f seconds.")
    var converged = false

    var cost = 0.0
    var iteration = 0
    val iterationStartTime = System.nanoTime()


    instr.foreach(_.logNumFeatures(numFeatures))
    val shouldComputeStats =
      DistanceMeasure.shouldComputeStatistics(centers.length)

    val shouldComputeStatsLocally =
      DistanceMeasure.shouldComputeStatisticsLocally(centers.length, numFeatures)

    val sc = data.sparkContext
    // Execute iterations of SOCCER algorithm until converged
    while (iteration < maxIterations && !converged) {
      val bcCenters = sc.broadcast(centers)
      val stats = if (shouldComputeStats) {
        if (shouldComputeStatsLocally) {
          Some(distanceMeasureInstance.computeStatistics(centers))
        } else {
          Some(distanceMeasureInstance.computeStatisticsDistributedly(sc, bcCenters))
        }
      } else {
        None
      }
      val bcStats = sc.broadcast(stats)

      val costAccum = sc.doubleAccumulator

      // Find the new centers
      val collected = data.mapPartitions { points =>
        val centers = bcCenters.value
        val stats = bcStats.value
        val dims = centers.head.vector.size

        val sums = Array.fill(centers.length)(Vectors.zeros(dims))

        // clusterWeightSum is needed to calculate cluster center
        // cluster center =
        //     sample1 * weight1/clusterWeightSum + sample2 * weight2/clusterWeightSum + ...
        val clusterWeightSum = Array.ofDim[Double](centers.length)

        points.foreach { point =>
          val (bestCenter, cost) = distanceMeasureInstance.findClosest(centers, stats, point)
          costAccum.add(cost * point.weight)
          distanceMeasureInstance.updateClusterSum(point, sums(bestCenter))
          clusterWeightSum(bestCenter) += point.weight
        }

        Iterator.tabulate(centers.length)(j => (j, (sums(j), clusterWeightSum(j))))
          .filter(_._2._2 > 0)
      }.reduceByKey { (sumweight1, sumweight2) =>
        axpy(1.0, sumweight2._1, sumweight1._1)
        (sumweight1._1, sumweight1._2 + sumweight2._2)
      }.collectAsMap()

      if (iteration == 0) {
        instr.foreach(_.logNumExamples(costAccum.count))
        instr.foreach(_.logSumOfWeights(collected.values.map(_._2).sum))
      }

      bcCenters.destroy()
      bcStats.destroy()

      // Update the cluster centers and costs
      converged = true
      collected.foreach { case (j, (sum, weightSum)) =>
        val newCenter = distanceMeasureInstance.centroid(sum, weightSum)
        if (converged &&
          !distanceMeasureInstance.isCenterConverged(centers(j), newCenter, epsilon)) {
          converged = false
        }
        centers(j) = newCenter
      }

      cost = costAccum.value
      instr.foreach(_.logNamedValue(s"Cost@iter=$iteration", s"$cost"))
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == maxIterations) {
      logInfo(s"SoccerKMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"SoccerKMeans converged in $iteration iterations.")
    }

    logInfo(s"The cost is $cost.")

    new KMeansModel(centers.map(_.vector), distanceMeasure, cost, iteration)
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
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   * @param seed Random seed for cluster initialization. Default is to generate seed based
   *             on system time.
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
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
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
