# cs838-spark-k-means-clustering

README:

AUTHOR: Zhenyuan Shen
E-MAIL: szywind@gmail.com


Before running the code:

In order to run this program on your environment, you may need to make the following modifications

(1) Line 11 os.environ['PYSPARK_PYTHON'] = sys.executable // set the python environment to include the dependencies like numpy.
(2) Line 27, 39: load the input files // you should transfer the input files "QuestionA1_data_6" and "stopwords" to your HDFS and modify the paths
(3) Line 79, 104: write to output files // you may modify the paths to save the model and results


Running the code:

USAGE: $SPARK_HOME/bin/spark-submit part_B.py
