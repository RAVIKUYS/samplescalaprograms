package com.learning.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark
import org.elasticsearch.spark
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.elasticsearch.spark.rdd.EsSpark 

object ESExample {
   def main(args:Array[String]){
    
    
      println("Creating spark context")
      //Creating a spark context and sql context
      val conf = new SparkConf().setAppName("Elastic Search Example").setMaster("local[*]")
      
      //define spark context object
      val sc = new SparkContext(conf)
      //Set the logger level to error
      sc.setLogLevel("ERROR")
      
      val sqlcont = new org.apache.spark.sql.SQLContext(sc)
       
      //Setting the elastic search context
      conf.set("es.auto.index.create", "true")
      
      
      var empRdd = sc.textFile("hdfs://127.0.0.1:54310/user/hduser/empinfo.csv")
      val headerRdd = empRdd.first()
      //Filtering the header row
      empRdd = empRdd.filter(x=>x!=headerRdd)
      val empInfoRdd = empRdd.map(x=>x.split(","))
      
      val empInfoSeq = empInfoRdd.collect().toSeq
      
      import org.elasticsearch._
      EsSpark.saveToEs(sc.makeRDD(empInfoSeq),"empinfo")
      
      
      sc.stop()
      
   }
}