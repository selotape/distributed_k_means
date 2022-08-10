package org.apache.spark.ml.clustering

import org.apache.spark.sql.DataFrame


class SoccerFormulae {

}

object SoccerFormulae {

  // TODO
  val MAX_ITERATIONS_DEFAULT = 123

  val DELTA_DEFAULT = 123.0
  val EPSILON_DEFAULT = 123.0
  val KPLUS_SCALER = 123
  val KPLUS_C = 1.23
  val MAX_SS_SIZE_C = 123
  val PHI_ALPHA_C = 123


  /** *
   * The allowed size of the "k+" clusters group
   */
  def kplus_formula(k: Int, dt: Double, ep: Double): Int = {
    KPLUS_SCALER * (k + KPLUS_C * math.log(1.1 * k / (dt * ep))).toInt
  }

  /** *
   * The size above which data doesn't fit inside a single machine,
   * so clustering must be distributed.
   */
  def max_subset_size_formula(n: Long, k: Int, ep: Double, dt: Double): Long = {
    (MAX_SS_SIZE_C * k * math.pow(n, ep) * math.log(1.1 * k / dt)).toInt
  }

  /** *
   * The probability to draw a datum into P1/P2 samples
   */
  def alpha_formula(n: Long, k: Int, ep: Double, dt: Double, N_current_size: Long): Long = {
    max_subset_size_formula(n, k, ep, dt) / N_current_size
  }

  /** *
   * Sum of distances of samples to their closest cluster center.
   */
  def risk_kmeans(N: DataFrame, C: DataFrame): DataFrame = {
    ???
    //    distances = pairwise_distances_argmin_min_squared(N, C)
    //    return np.sum(distances)
  }

  def risk_truncated(P2: DataFrame, C: DataFrame, r: Int): Double = {
    ???
    //      distances = pairwise_distances_argmin_min_squared(P2, C)
    //      distances.sort()
    //
    //      if r >= len(P2):
    //          return 0  # The "trivial risk"
    //
    //      return np.sum(distances[:len(distances) - r])
  }

  //risk = risk_kmeans

  def phi_alpha_formula(alpha: Double, k: Int, dt: Double, ep: Double): Double = {
    (PHI_ALPHA_C / alpha) * math.log(1.1 * k / (dt * ep))
  }

  def r_formula(alpha: Double, k: Int, phi_alpha: Double): Int = {
    (1.5 * alpha * (k + 1) * phi_alpha).toInt
  }

  def v_formula(psi: Double, k: Int, phi_alpha: Double): Double = {
    psi / (k * phi_alpha)
  }

  def pairwise_distances_argmin_min_squared(X: DataFrame, Y: DataFrame): DataFrame = {
    ???
    //      linear_dists = pairwise_distances_argmin_min(X, Y)[1]
    //    square_dists = np.square(linear_dists)
    //    return square_dists
  }

  def alpha_s_formula(k: Int, n: Int, ep: Double, len_R: Int): Double = {
    9 * k * (math.pow(n, ep) * math.log(n)) / len_R
  }

  def alpha_h_formula(n: Int, ep: Double, len_R: Int): Double = {
    4 * math.pow(n, ep) * math.log(n) / len_R
  }

  //def Select(S, H, n):
  //    dists: np.ndarray = pairwise_distances_argmin_min_squared(H, S)
  //    dists.sort()
  //
  //    if len(dists) < 8 * log(n):
  //        logging.warning("len(dists) < 8*log(n) ☹️💔")
  //        return dists[0]
  //
  //    return dists[int(-8 * log(n))]
  //
  //
  //def measure_weights(N, C):
  //    chosen_centers = pairwise_distances_argmin(N, C)
  //    center_weights = np.zeros((len(C),), dtype=np.intc)
  //    for cc in chosen_centers:
  //        center_weights[cc] += 1
  //    return center_weights

}