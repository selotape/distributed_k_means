import pandas as pd
import numpy as np
from random import choice

from sklearn.metrics import pairwise_distances_argmin_min
from k_median_clustering.math import Blackbox, risk, distance
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


def scalable_k_means_clustering(N: pd.DataFrame, iterations: int, l: int):
    C = pd.DataFrame().append(N.iloc[[choice(range(len(N)))]])
    psii = risk(N, C)
    for i in range(iterations):
        calcs = pd.DataFrame()

        calcs['dists'] = np.square(pairwise_distances_argmin_min(N, C, metric=distance)[1])
        calcs['probs'] = (calcs['dists']) / psii
        # TODO - figure out why this didnt work -  calcs['coin_toss'] = np.random.uniform(size=len(N), )
        draws = np.random.choice(len(N), l, p=calcs['probs'], replace=False)
        C = C.append(N.iloc[draws])
        psii = risk(N, C)

    return C
