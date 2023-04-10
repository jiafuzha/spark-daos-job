Please check the wiki page, https://wiki.jlse.anl.gov/pages/viewpage.action?spaceKey=inteldga&title=Getting+Started+with+Spark+Job+Submit+Tool+On+Sunspot for how to run the tool.


OLD:

1. copy all config files under "conf-to-your-submit-dir" to your working directory where you run submission script.

2. run "/path-to-submit-script/bin/submit-spark.sh -h" to see how to run your Spark job in different ways.

3. change to your working directory and run the submission script. For example, you can run Dense KMeans example with below command,

	/soft/storage/daos/spark/spark-daos-job/bin/submit-spark.sh -t 30 -n 2 -q arcticus /soft/storage/daos/spark/spark-daos-job/example/dense_kmeans_example.py

   The above command assumes the dataset in "libsvm" format is put to DAOS path "daos://pool0/cont1/kmeans/input/libsvm".


4. check README.docx for script details.
