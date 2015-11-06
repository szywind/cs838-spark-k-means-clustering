from pyspark import mllib
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import IDF, HashingTF
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import os
import sys
import re

os.environ['PYSPARK_PYTHON'] = sys.executable

## Step 0: Set up the cluster configurations
conf = (SparkConf().set("spark.master","spark://10.0.1.105:7077")\
	           .set("spark.eventLog.enabled)",True)\
	           .set("spark.task.cpus",1)\
	           .set("spark.driver.memory","1g")\
	           .set("spark.executor.cores","4")\
	           .set("spark.executor.memory","21000m")\
	           .set("spark.eventLog.dir","/home/ubuntu/storage/logs")\
	           .setAppName("CS-838-Assignment3-Question2"))
sc = SparkContext(conf=conf)

nPartition = 1000
nFeature = 50000
## Step 1: Load documents (one per line) and parse the data
file = 'hdfs://10.0.1.105:8020/assignment3/QuestionA1_data_6'
#file = 'hdfs://10.0.1.105:8020/assignment3/tweets'
#file = 'tweets.txt'

## Step 2: Word Cleaning and Processing
#documents = sc.textFile(file).map(lambda line: line.split())
def processTweet(line):
	line = re.sub("[^A-Za-z']+", ' ', line)
	line = line.replace('http', ' ')
	line = line.replace('https', ' ')
	line = line.lower()
	return line.strip().split()
	
documents = sc.textFile(file, nPartition).map(lambda line: processTweet(line))

stopWordsFile= 'hdfs://10.0.1.105:8020/assignment3/stopwords'
#stopWordsFile = "stopWords"
stopwordsList = sc.textFile(stopWordsFile).collect()

def filterStopWords(x):
	filtered_x = []
	for word in x:
		if word not in stopwordsList and len(word)>1:
			filtered_x.append(word)
	return filtered_x

documents = documents.map(lambda x: filterStopWords(x)).filter(lambda x: len(x)>0)


## Step 3: Extract TF-IDF features
hashingTF = HashingTF(nFeature)   # default is 2^20
tf = hashingTF.transform(documents)
tf.cache()
idf = IDF(minDocFreq=5).fit(tf)
#idf = IDF().fit(tf)
tfidf = idf.transform(tf).repartition(nPartition)
tf.unpersist()
del idf
tfidf.cache()

## Step 4: Clustering with k-mean algorithm

#pool = [10, 100, 1000]

pool = [1000]
for nCluster in pool:
	# Build the model (cluster the data)
	kmeans_model = KMeans.train(tfidf, nCluster, maxIterations=10, runs=1, initializationMode="random")

	# Evaluate clustering by computing Within Set Sum of Squared Errors
	'''
	def error(point):
	    center = kmeans_model.centers[kmeans_model.predict(point)]
	    return sqrt(sum([x**2 for x in (point - center)]))

	WSSSE = tfidf.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	print("Within Set Sum of Squared Error = " + str(WSSSE))
	'''
	# Save the model
	#kmeans_model.save(sc, "km_model_" + str(nCluster))
	kmeans_model.save(sc, "hdfs://10.0.1.105:8020/assignment3/km_model_50K_"+str(nCluster))

	## Step 5: Prediction
#nCluster = 10
	#kmeans_model = KMeansModel.load(sc, "hdfs://10.0.1.105:8020/assignment3/km_model_"+str(nCluster))
#kmeans_model = KMeansModel.load(sc, "km_model_" + str(nCluster))
#kmeans_model.centers()

	def dataClusterPairs(data, cluster_id):
		result = []
		for i in xrange(len(data)):
			result.append([cluster_id[i], data[i]])
		return result

	cluster_id = kmeans_model.predict(tfidf).collect()
	data = documents.collect()
	mydocuments = sc.parallelize(dataClusterPairs(data, cluster_id), nPartition)
	documents.unpersist()

	## Step 6: Find the topics of each clusters by counting the top 5 most frequent words
	clustered_data = mydocuments.reduceByKey(lambda x,y: x+y)#.sortByKey()
	clustered_data = clustered_data.map(lambda (id, wordLists): wordLists).collect()

	topics = []
	for i in clustered_data:
		word_count = sc.parallelize(i, nPartition)
		topics.append(word_count.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).takeOrdered(10, key=lambda x: -x[1]))

	## Step 7: Save the result to hdfs
	result_rdd = sc.parallelize(topics)
	result_rdd.saveAsTextFile('hdfs://10.0.1.105:8020/assignment3/result_partB_50K_'+str(nCluster))

