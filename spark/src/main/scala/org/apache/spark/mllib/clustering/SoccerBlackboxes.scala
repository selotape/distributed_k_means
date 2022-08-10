package org.apache.spark.mllib.clustering

import org.apache.spark.mllib.clustering.{DistanceMeasure, KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

class SoccerBlackboxes {

}

object SoccerBlackboxes {

  /***
   * Runs the blackbox offline clustering algorithm
   * @return the k chosen clusters
   * TODO - support multiple blackboxii
   */
//  def A(N: DataFrame, k: Int): Array[VectorWithNorm] = {
//    val kmeans = new KMeans().setK(k) // TODO - support seeds
//    val model = kmeans.fit(N)
//    model.clusterCenters.map(v=> new VectorWithNorm(v)) // TODO - make sure this is cheap
//  }

}
