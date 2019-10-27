import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql


object models {

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("Usage: <input path> <output path>")
    }
    val spark = SparkSession.builder().appName("tweetClassification").getOrCreate()

    import spark.implicits._
    val sc = spark.sparkContext
    var output = ""

    val df = spark.read.option("multiLine", "true").json(args(0))
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val indexer = new StringIndexer()
      .setInputCol("stars")
      .setOutputCol("label")
      .setHandleInvalid("keep")

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("Non-Stop")

    val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")


    //    Logistic regression
    val lr = new LogisticRegression()
      .setMaxIter(15)

    val lrpipeline = new Pipeline()
      .setStages(Array(indexer, tokenizer, remover, hashingTF, lr))

    val lrparamGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.maxIter, Array(15, 20))
      .addGrid(lr.elasticNetParam, Array(0.8, 0.5, 0.1))
      .addGrid(lr.threshold, Array(0.1, 0.5, 0.8))
      .build()


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val lrcv = new CrossValidator()
      .setEstimator(lrpipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(lrparamGrid)
      .setNumFolds(5)

    val lrModel = lrcv.fit(trainingData)


    val lrTesting = lrModel.transform(testData)
    val lrAccuracy = evaluator.evaluate(lrTesting)
    println("Logistic regression")
    output += "Accuracy of Logistic Regression: " + lrAccuracy + "\n"


    //  Randomforestclassifier

    val rf = new RandomForestClassifier()
      .setLabelCol("label")

    val rfpipeline = new Pipeline()
      .setStages(Array(indexer, tokenizer, remover, hashingTF, rf))



    val rfparamGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(rf.maxDepth, Array(10, 15, 20))
      .addGrid(rf.numTrees, Array(20, 35, 50))
      .build()


    val rfCV = new CrossValidator()
      .setEstimator(rfpipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(rfparamGrid)
      .setNumFolds(5)


    val rfmodel = rfCV.fit(trainingData)
    val rftesting = rfmodel.transform(testData)
    val rfTestAcc = evaluator.evaluate(rftesting)

    output += "Accuracy of Randomforestclassifier:" + rfTestAcc + "\n"


    //     GBT classifier

    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    val gbtpipeline = new Pipeline()
      .setStages(Array(indexer, tokenizer, remover, hashingTF, gbt))

    val gbtParamGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000, 5000))
      .addGrid(gbt.maxIter, Array(200, 400))
      .addGrid(gbt.maxDepth, Array(20, 35, 50))
      .build()

    val gbtCV = new CrossValidator()
      .setEstimator(gbtpipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(gbtParamGrid)
      .setNumFolds(5)

    val gbtmodel = gbtCV.fit(trainingData)
    val gbttesting = gbtmodel.transform(testData)
    val gbtTestAcc = evaluator.evaluate(gbttesting)


    output += "Accuracy of GBTclassifier:" + gbtTestAcc + "\n"

    sc.parallelize(List(output)).saveAsTextFile(args(1))
    sc.stop()

  }

}
