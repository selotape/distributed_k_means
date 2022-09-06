package org.apache.spark.mllib.clustering


class SoccerFormulae {

}

object SoccerFormulae {

  // TODO
  val MAX_ITERATIONS_DEFAULT = 123

  val DELTA_DEFAULT = 0.1
  val PHI_ALPHA_C = 6.5
  val MAX_SS_SIZE_C = 36
  val KPLUS_C = 9
  val KPLUS_SCALER = 1

  val L_TO_K_RATIO = 2.0
  val INNER_BLACKBOX_L_TO_K_RATIO = 2
  val KMEANS_INIT_MODE = KMeans.RANDOM // "K_MEANS_PARALLEL"


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
  def max_subset_size_formula(n: Long, k: Int, ep: Double, dt: Double): Double = {
    MAX_SS_SIZE_C * k * math.pow(n, ep) * math.log(1.1 * k / dt)
  }

  /** *
   * The probability to draw a datum into P1/P2 samples
   */
  def alpha_formula(n: Long, k: Int, ep: Double, dt: Double, N_current_size: Long): Double = {
    val alpha = max_subset_size_formula(n, k, ep, dt) / N_current_size
    require(alpha > 0 & alpha < 0.5, s"alpha must be in [0, 0.5)") // TODO - verify With hess this is fine.
    alpha
  }

  def phi_alpha_formula(alpha: Double, k: Int, dt: Double, ep: Double): Double = {
    (PHI_ALPHA_C / alpha) * math.log(1.1 * k / (dt * ep)) // TODO - add assertions
  }

  def r_formula(alpha: Double, k: Int, phi_alpha: Double): Int = {
    (1.5 * alpha * (k + 1) * phi_alpha).toInt
  }

  def v_formula(psi: Double, k: Int, phi_alpha: Double): Double = {
    psi / (k * phi_alpha)
  }

  def alpha_s_formula(k: Int, n: Int, ep: Double, len_R: Int): Double = {
    9 * k * (math.pow(n, ep) * math.log(n)) / len_R
  }

  def alpha_h_formula(n: Int, ep: Double, len_R: Int): Double = {
    4 * math.pow(n, ep) * math.log(n) / len_R
  }

  def elapsedSecs(startTimeMillis: Long): Double = {
    // TODO - move to a static utils companion object
    (System.currentTimeMillis() - startTimeMillis) / 1000.0
  }

}