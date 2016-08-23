# Spark Machine learning sample using dataframe and ML

Spark machine learning library used to be mllib. Then a new machine learning library ml is introduced and from Spark 2.0 it will be the default machine learning library. The new ml library works differently than the old mllib. It supports dataframe and the new pipeline. In the following we will examine how the new library works by demonstrate a few simple algorithms and also create one simple pipeline. By comparin this with how mllib works, you could tell the difference between the two.

## Setup environment 
Note: this is same as the RDD and mllib, if you have done this already, skip to machine learning process

Download Hortonworks HDP 2.4 [sandbox](https://hortonworks.com/downloads/#sandbox) and prepare the sandbox following this [instruction](http://hortonworks.com/hadoop-tutorial/learning-the-ropes-of-the-hortonworks-sandbox/)  

After the sandbox is prepared, you should see in Ambari dashboard that Spark and Zeppelin Notebook both turns green

THen go to Zeppelin notebook using [http://127.0.0.1:9995](http://127.0.0.1:9995). 

Click Notebook -> create new notebook, name the notebook. 

##Prepare the flight delay dataset
### Download dataset
Download the dataset from internet
```
%sh
wget http://stat-computing.org/dataexpo/2009/2007.csv.bz2 -O /tmp/flights_2007.csv.bz2
wget http://stat-computing.org/dataexpo/2009/2008.csv.bz2 -O /tmp/flights_2008.csv.bz2
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2007.csv.gz -O /tmp/weather_2007.csv.gz
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2008.csv.gz -O /tmp/weather_2008.csv.gz
ls -l /tmp/
```
### Copy dataset to HDFS
```
%sh
#remove existing copies of dataset from HDFS
hdfs dfs -rm -r -f /tmp/airflightsdelays
hdfs dfs -mkdir /tmp/airflightsdelays

#put data into HDFS
hdfs dfs -put /tmp/flights_2007.csv.bz2 /tmp/flights_2008.csv.bz2 /tmp/airflightsdelays/
hdfs dfs -put /tmp/weather_2007.csv.gz /tmp/weather_2008.csv.gz /tmp/airflightsdelays/
hdfs dfs -ls -h /tmp/airflightsdelays
```
## Machine learning process
###Prepare the traning and testing data 
```
%spark
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.attribute.NominalAttribute

//calculate minuted from midnight, input is military time format 
def getMinuteOfDay(depTime: String) : Int = (depTime.toInt / 100).toInt * 60 + (depTime.toInt % 100)

//this is needed for Decision Tree classifier
val meta = NominalAttribute.defaultAttr.withName("label").withValues("0.0", "1.0").toMetadata

val flight2007 = sc.textFile("/tmp/airflightsdelays/flights_2007.csv.bz2")
val header = flight2007.first

val trainingData = flight2007
                    .filter(x => x != header)
                    .map(x=>x.split(","))
                    .filter(x => x(21) == "0")
                    .filter(x => x(16) == "ORD")
                    .filter(x => x(14) != "NA")
                    .map(x => (if (x(14).toInt >= 15) 1.0 else 0.0, Vectors.dense(x(1).toInt, x(2).toInt, x(3).toInt, getMinuteOfDay(x(4)), getMinuteOfDay(x(6)), x(11).toInt, x(15).toInt, x(18).toInt)))
                    .toDF("label", "features")
                    .withColumn("label", $"label".as("label", meta))
                    
trainingData.cache

val flight2008 = sc.textFile("/tmp/airflightsdelays/flights_2008.csv.bz2")
val testingData = flight2008
                    .filter(x => x != header)
                    .map(x=>x.split(","))
                    .filter(x => x(21) == "0")
                    .filter(x => x(16) == "ORD")
                    .filter(x => x(14) != "NA")
                    .map(x => (if (x(14).toInt >= 15) 1.0 else 0.0, Vectors.dense(x(1).toInt, x(2).toInt, x(3).toInt, getMinuteOfDay(x(4)), getMinuteOfDay(x(6)), x(11).toInt, x(15).toInt, x(18).toInt)))
                    .toDF("label", "features")
                    .withColumn("label", $"label".as("label", meta))
testingData.cache
```
###Train model with Decision Tree 
```
%spark
import org.apache.spark.ml.classification.DecisionTreeClassifier

// Build the Decision Tree model
val decisionTree = new DecisionTreeClassifier().setMaxDepth(10).setMaxBins(100).setImpurity("gini")
val decisionTreeModel = decisionTree.fit(trainingData)

decisionTreeModel.transform(testingData)
```
###Train model with Random Forest 
```
%spark
import org.apache.spark.ml.classification.RandomForestClassifier

// Build the Random Forest model
val randomForest = new RandomForestClassifier().setNumTrees(100).setFeatureSubsetStrategy("auto")
val randomForestModel = randomForest.fit(trainingData)

randomForestModel.transform(testingData)
```

###Normalize data for regression algorithms
```
%spark
import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true);

val trainingScalerModel = scaler.fit(trainingData);
val scaledTrainingData = trainingScalerModel.transform(trainingData);

val testingScalerModel = scaler.fit(testingData);
val scaledTestingData = testingScalerModel.transform(testingData);
```
###Train model with Logistic Regression 
```
%spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.ml.Pipeline

// Create a LogisticRegression instance.  This instance is an Estimator.
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01).setFeaturesCol("scaledFeatures");
val pipeline = new Pipeline().setStages(Array(scaler, lr));
val model = pipeline.fit(trainingData);
model.transform(testingData);
```
