/*Pls. note this is not my proprietary code, I am trying different examples provided by multiple people */
package com.learning.scala

import org.apache.spark.sql.SparkSession

object sparkMLExample {
  case class trainClass(
      PassengerId:Float,Survived:Float,Pclass:Float,
      Name:String,Sex:String,Age:Int,SibSp:Int,
      Parch:Int,Ticket:String,Fare:Float,
      Cabin:String,Embarked:String)
      
  case class trainSubsetClass
  (Survived:Float,Pclass:Float,
      Sex:String,Age:Int,
      Fare:Float,
     Embarked:String)
  
  def main(args:Array[String]){
    
    
    println("Creating spark context")
    //Creating a spark context and sql context
    val sparkSession = SparkSession.builder().master("local[*]")
        .appName("Spark ML Example")
        .enableHiveSupport()
        .getOrCreate()
    val sqlcont = sparkSession.sqlContext
    
    
      import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType}
      val trainschema = StructType(Array(
            StructField("PassengerId", FloatType, true),
            StructField("Survived", FloatType, true),
            StructField("Pclass", FloatType, true),
            StructField("Name", StringType, true),
            StructField("Sex", StringType,true),
            StructField("Age", IntegerType, true),
            StructField("SibSp", IntegerType, true),
            StructField("Parch", IntegerType, true),
            StructField("Ticket", StringType, true),
            StructField("Fare", FloatType,true),
            StructField("Cabin", StringType, true),
            StructField("Embarked", StringType, true)));
    
    //Get train CSV file from Kaggle and read the file from HDFS file system
    import sparkSession.implicits._
    val trainDataSet = sqlcont.read.format("csv").option("header","true").option("inferSchema","false")
      .schema(trainschema)
      .load("hdfs://127.0.0.1:54310/user/hduser/train.csv")
      .as[trainClass]
      
    //Displaying 5 records 
    trainDataSet.show(5,false)
    var trainDS  = trainDataSet.select($"Survived",$"Pclass",$"Sex",$"Age",$"Fare",$"Embarked").as[trainSubsetClass]
    
    //Printing first 10 rows of train SubsetClass
    trainDS.show(10,false)
    
   
    //Dropping null values from trainDS
    println("Count of records before dropping records with null values " + trainDS.count)
    trainDS.na.drop()
    println("Count of records after dropping records with null values " + trainDS.count)
    
    import org.apache.spark.ml.feature.{StringIndexer,VectorAssembler}
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("Gender")
    val boardedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("Boarded")
    //Adding the new indexer cols to dataset
    var indexedDS = genderIndexer.fit(trainDS).transform(trainDS)
    indexedDS = boardedIndexer.fit(indexedDS).transform(indexedDS)
    indexedDS.show(20,false)
    
    //Dropping unwanted columns
    indexedDS = indexedDS.drop("Sex")
    indexedDS = indexedDS.drop("Embarked")
    indexedDS.show(20,false)
    indexedDS.na.drop()
    
    //Creating a vector assembler to store all features except survived in a separate column.
    //This is the most important step in Machine Learning
    val required_features = Array("Pclass","Age","Fare","Gender","Boarded")
    val vectorAssembler = new VectorAssembler().setInputCols(required_features).setOutputCol("features").setHandleInvalid("skip")
    //Stopping the spark context
    val transformedDS = vectorAssembler.transform(indexedDS)
    transformedDS.show(20,false)
    
    
    //Splitting dataset into 2 parts before applying estimator
    val divData = transformedDS.randomSplit(Array(0.8,0.2), 13L)
    val trainingDataset = divData(0)
    val testingDataset = divData(1)
    
    import org.apache.spark.ml.classification.RandomForestClassifier
    val rf = new RandomForestClassifier()
      .setFeaturesCol(vectorAssembler.getOutputCol)
      .setLabelCol("Survived")
      .setNumTrees(20)
      .setMaxDepth(5)
      .setMaxBins(32)
      .setMinInstancesPerNode(1)
      .setMinInfoGain(0.0)
      .setCacheNodeIds(false)
      .setCheckpointInterval(10)
    
    val model = rf.fit(trainingDataset)
    val predictions = model.transform(testingDataset)
    
    //Evaluating the model for accuracy
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    
    
    
    // Select (prediction, true label) and compute test error
    val accuracyEval = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    
    
    val accuracyCalc = accuracyEval.evaluate(predictions)
    print("Test Accuracy = " + accuracyCalc)
      
    //Stopping the spark session
    sparkSession.stop() 
  }
}