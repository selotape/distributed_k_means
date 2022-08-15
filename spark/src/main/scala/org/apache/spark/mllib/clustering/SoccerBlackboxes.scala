package org.apache.spark.mllib.clustering

import org.apache.spark.rdd.RDD

class SoccerBlackboxes {

}

object SoccerBlackboxes {

  /** *
   * Runs the blackbox offline clustering algorithm
   *
   * @return the k chosen clusters
   *         TODO - support multiple blackboxii
   */
  //  def A(N: DataFrame, k: Int): Array[VectorWithNorm] = {
  //    val kmeans = new KMeans().setK(k) // TODO - support seeds
  //    val model = kmeans.fit(N)
  //    model.clusterCenters.map(v=> new VectorWithNorm(v)) // TODO - make sure this is cheap
  //  }

  def A_final(centers: RDD[VectorWithNorm], k: Int, center_weights: Array[Double]): Array[VectorWithNorm] = {
    centers.take(k)
  }

}
