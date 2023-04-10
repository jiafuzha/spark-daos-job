import sys
import traceback
from time import time
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import split, col
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import array_to_vector


defaultUri = "daos://pool0/cont1/kmeans/input/libsvm"
defaultK = 300
defaultIters = 40
defaultInitMode = "random"
helpMsg = "\nParameter's order is,\n" + \
          f"1, datset URI (default: {defaultUri} )\n" + \
          f"2, k (default: {defaultK} )\n" + \
          f"3, max iteration (default: {defaultIters} )\n" + \
          f"4, init mode (default: {defaultInitMode}, Options: Random or Parallel )\n"

print("Running 'Dense KMeans Example - HiBench Dataset:\n")
params = ' '.join([e for e in sys.argv])
print(sys.argv)
print("\n")

def main():
    uriStr = defaultUri
    k = defaultK
    iters = defaultIters
    initMode = defaultInitMode
    spark = SparkSession.builder.appName("Dense KMeans Example - HiBench Dataset, " + params).getOrCreate()
    try:
        argLen = len(sys.argv)
        uriStr = defaultUri
        if argLen > 1 :
            uriStr = sys.argv[1]
        if argLen > 2 :
            k = int(sys.argv[2])
        if argLen > 3 :
            iters = int(sys.argv[3])
        if argLen > 4 :
            initMode = sys.argv[4]

#        df = spark.read.format("libsvm").load(uriStr)
        df = spark.read.option("quote", " ").csv(uriStr).toDF("features")
        df = df.select(split(col("features"), ",").alias("features"))
        df = df.withColumn("features", df.features.cast("array<double>"))
        df = df.withColumn("features", array_to_vector("features"))
        df.cache()
        df.count()


        kmeans = KMeans()
        kmeans.setK(k)
        kmeans.setMaxIter(iters)
        kmeans.setInitMode("k-means||" if (initMode=="Parallel") else "random")
        kmeans.setSeed(777)
        kmeans.setTol(0)
        
        startMill = int(time() * 1000)
        model = kmeans.fit(df)
        print("kmeans fit took time (ms) = %d\n" % (int(time() * 1000) - startMill))
        # Make predictions
        predictions = model.transform(df)

        # Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator()

        silhouette = evaluator.evaluate(predictions)
        print("Silhouette with squared euclidean distance = " + str(silhouette))

        # Shows the result.
        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)

        print("Kmeans end took time (ms) = %d\n" % (int(time() * 1000) - startMill))
    # $example off$
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise Exception(helpMsg) from e
    finally:
       spark.stop()


if __name__ == "__main__":
    main()
