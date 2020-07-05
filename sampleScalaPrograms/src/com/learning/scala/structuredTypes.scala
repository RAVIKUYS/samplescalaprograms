package com.learning.scala

import org.apache.spark.sql.SparkSession


object structuredTypes {
  
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
    val sparkSession = SparkSession.builder().master("local[*]").appName("structured types example").getOrCreate()
    val sqlcont = sparkSession.sqlContext
    
    
      import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType}
      val trainschema = StructType(Array(
            StructField("PassengerId", FloatType, true),
            StructField("Survived", FloatType, true),
            StructField("Pclass", FloatType, true),
            StructField("Name", FloatType, true),
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
    //Stopping the spark context
    sparkSession.stop() 
  }
}