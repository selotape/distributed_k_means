from k_median_clustering.math import Blackbox, risk

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession


def spark_kmeans(N, k):
    spark = SparkSession.builder.getOrCreate()
    dataset = spark.createDataFrame(N)
    kmeans = KMeans().setK(k)
    model = kmeans.fit(dataset)
    predictions = model.transform(dataset)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    return silhouette


def blackbox(N, k):
    kmeans = Blackbox(n_clusters=k, n_jobs=-1).fit(N)
    return risk(N, kmeans.cluster_centers_)
