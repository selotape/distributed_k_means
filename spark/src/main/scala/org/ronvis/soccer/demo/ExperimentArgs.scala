package org.ronvis.soccer.demo

import org.rogach.scallop._


class ExperimentArgs(arguments: Seq[String]) extends ScallopConf(arguments) {
  val dataset = opt[String](required = false, default = Option("HIGGS_top20k")) // TODO - use this
  val k = opt[Int](required = false, default = Option(25))
  val m = opt[Int](required = false, default = Option(4))
  val maxIters = opt[Int](required = false, default = Option(3))
  val seed = opt[Int](required = false, default = Option(1))
  val delta = opt[Double](required = false, default = Option(.1))
  val epsilon = opt[Double](required = false, default = Option(.05))

  verify()
}
