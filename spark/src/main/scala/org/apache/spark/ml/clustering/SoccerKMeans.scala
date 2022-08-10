/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.clustering

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.functions.checkNonNegativeWeight
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.mllib.clustering.MLlibSoccerKMeans
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel


/**
 * K-means clustering with support for k-means|| initialization proposed by Bahmani et al.
 *
 * @see <a href="https://doi.org/10.14778/2180912.2180915">Bahmani et al., Scalable k-means++.</a>
 */
@Since("1.5.0")
class SoccerKMeans @Since("1.5.0")(
                                @Since("1.5.0") override val uid: String)
  extends Estimator[KMeansModel] with KMeansParams with DefaultParamsWritable {

  @Since("1.5.0")
  override def copy(extra: ParamMap): SoccerKMeans = defaultCopy(extra)

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("kmeans"))

  /** @group setParam */
  @Since("1.5.0")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setK(value: Int): this.type = set(k, value)

  /**
   * The number of clusters to create (k). Must be &gt; 1. Note that it is possible for fewer than
   * k clusters to be returned, for example, if there are fewer than k distinct points to cluster.
   * Default: 2.
   * @group param
   */
  @Since("1.5.0")
  final val m = new IntParam(this, "k", "The number of available machines. " +
    "Must be >= 1.", ParamValidators.gt(0))

  /** @group getParam */
  @Since("1.5.0")
  def getM: Int = $(m)


  /** @group setParam */
  @Since("1.5.0")
  def setM(value: Int): this.type = set(m, value)

  /** @group expertSetParam */
  @Since("1.5.0")
  def setInitMode(value: String): this.type = set(initMode, value)

  /** @group expertSetParam */
  @Since("2.4.0")
  def setDistanceMeasure(value: String): this.type = set(distanceMeasure, value)

  /** @group expertSetParam */
  @Since("1.5.0")
  def setInitSteps(value: Int): this.type = set(initSteps, value)

  /** @group setParam */
  @Since("1.5.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  @Since("1.5.0")
  def setTol(value: Double): this.type = set(tol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  @Since("3.0.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): KMeansModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, featuresCol, predictionCol, k, initMode, initSteps, distanceMeasure,
      maxIter, seed, tol, weightCol)
    val algo = new MLlibSoccerKMeans()
      .setK($(k))
      .setM($(m))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setEpsilon($(tol))
      .setDistanceMeasure($(distanceMeasure))

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      checkNonNegativeWeight(col($(weightCol)).cast(DoubleType))
    } else {
      lit(1.0)
    }
    val instances = dataset.select(DatasetUtils.columnToVector(dataset, getFeaturesCol), w)
      .rdd.map { case Row(point: Vector, weight: Double) => (OldVectors.fromML(point), weight) }

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    val parentModel = algo.runWithWeight(instances)
    val model = copyValues(new KMeansModel(uid, parentModel).setParent(this))

    val summary = new KMeansSummary(
      model.transform(dataset),
      $(predictionCol),
      $(featuresCol),
      $(k),
      parentModel.numIter,
      parentModel.trainingCost)

    model.setSummary(Some(summary))
    instr.logNamedValue("clusterSizes", summary.clusterSizes)
    model
  }

  @Since("1.5.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}


// TODO - uncomment and fix
//@Since("1.6.0")
//object SoccerKMeans extends DefaultParamsReadable[SoccerKMeans] {
//
//  @Since("1.6.0")
//  override def load(path: String): SoccerKMeans = super.load(path)
//}
//
//
