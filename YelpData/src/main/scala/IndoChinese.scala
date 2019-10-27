import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.DistributedLDAModel


object IndoChinese {
  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("Usage: <Review data path> <Business data path> <Output path>")
    }


    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

    import spark.implicits._

    val reviewData = spark.read.option("multiLine", "true").json(args(0)).filter($"text" =!= "")
    var businessData = spark.read.option("header","true")
      .json(args(1)).withColumnRenamed("business_id","business_id2")
      .withColumnRenamed("stars","notusedstars")
    val allReviews = reviewData.join(businessData, reviewData
      .col("business_id") === businessData.col("business_id2"),"left")

    val IndianRest = allReviews.select("*").where(col("categories")
      .contains("Indian") && col("categories").contains("Restaurants"))
    val ChineseRest = allReviews.select("*").where(col("categories")
      .contains("Chinese") && col("categories").contains("Restaurants"))

    var output=""
    output += "Indian Restaurants count: " + IndianRest.count() + "\n"
    output += "Chinese Restaurants count: " + ChineseRest.count() + "\n"

    val topIndian = IndianRest.groupBy("business_id").avg("stars").orderBy(desc("avg(stars)"))
    val worstIndian = IndianRest.groupBy("business_id").avg("stars").orderBy(asc("avg(stars)"))

    val topChinese = ChineseRest.groupBy("business_id").avg("stars").orderBy(desc("avg(stars)"))
    val worstChinese = ChineseRest.groupBy("business_id").avg("stars").orderBy(asc("avg(stars)"))

    output += "Top Indian restaurant" + "\n" + topIndian.collectAsList().get(0).mkString("\n") + "\n"
    output += "Worst Indian restaurant" + "\n" + worstIndian.collectAsList().get(0).mkString("\n") + "\n"

    output += "Top Chinese restaurant" + "\n" + topChinese.collectAsList().get(0).mkString("\n") + "\n"
    output += "Worst Chinese restaurant" + "\n" + worstChinese.collectAsList().get(0).mkString("\n") + "\n"

    val topIndianBusiness = topIndian.collectAsList().get(0).getString(0)
    val worstIndianBusiness = worstIndian.collectAsList().get(0).getString(0)

    val topChineseBusiness = topChinese.collectAsList().get(0).getString(0)
    val worstChineseBusiness = worstChinese.collectAsList().get(0).getString(0)

    val topIndianBusinessData = reviewData.filter(col("business_id") === topIndianBusiness)
    val worstIndianBusinessData = reviewData.filter(col("business_id") === worstIndianBusiness)

    val topChineseBusinessData = reviewData.filter(col("business_id") === topChineseBusiness)
    val worstChineseBusinessData = reviewData.filter(col("business_id") === worstChineseBusiness)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")
    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(2048)
    val lda = new LDA()
      .setK(2)
      .setMaxIter(50)
      .setOptimizer("em")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorizer, lda))

    val topIndianModel= pipeline.fit(topIndianBusinessData)
    val worstIndianModel = pipeline.fit(worstIndianBusinessData)

    val topChineseModel= pipeline.fit(topChineseBusinessData)
    val worstChineseModel = pipeline.fit(worstChineseBusinessData)

    val vectorizerIndianTop = topIndianModel.stages(2).asInstanceOf[CountVectorizerModel]
    val vectorizerChineseTop = topChineseModel.stages(2).asInstanceOf[CountVectorizerModel]

    val ldaIndianTop = topIndianModel.stages(3).asInstanceOf[DistributedLDAModel]
    val ldaChineseTop = topChineseModel.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabListIndianTop = vectorizerIndianTop.vocabulary
    val termsIdx2StrIndianTop = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabListIndianTop(idx)) }

    val vocabListChineseTop = vectorizerChineseTop.vocabulary
    val termsIdx2StrChineseTop = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabListChineseTop(idx)) }

    val IndianTopics = ldaIndianTop.describeTopics(maxTermsPerTopic = 15)
      .withColumn("terms", termsIdx2StrIndianTop(col("termIndices")))
    val ChineseTopics = ldaChineseTop.describeTopics(maxTermsPerTopic = 15)
      .withColumn("terms", termsIdx2StrChineseTop(col("termIndices")))

    val IndianRow = IndianTopics.select("topic", "terms", "termWeights").collectAsList()
    output += "\n"+ IndianRow.toArray.mkString(" ")

    val ChineseRow = ChineseTopics.select("topic", "terms", "termWeights").collectAsList()
    output += "\n"+ ChineseRow.toArray.mkString(" ")

    val vectorizerIndianWorst = worstIndianModel.stages(2).asInstanceOf[CountVectorizerModel]
    val vectorizerChineseWorst = worstChineseModel.stages(2).asInstanceOf[CountVectorizerModel]

    val ldaIndianWorst = worstIndianModel.stages(3).asInstanceOf[DistributedLDAModel]
    val ldaChineseWorst = worstChineseModel.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabListIndianWorst = vectorizerIndianWorst.vocabulary
    val termsIdx2StrIndianWorst = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabListIndianWorst(idx)) }

    val vocabListChineseWorst = vectorizerChineseWorst.vocabulary
    val termsIdx2StrChineseWorst = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabListChineseWorst(idx)) }

    val IndianTopics1 = ldaIndianWorst.describeTopics(maxTermsPerTopic = 15)
      .withColumn("terms", termsIdx2StrIndianWorst(col("termIndices")))
    val ChineseTopics1 = ldaChineseWorst.describeTopics(maxTermsPerTopic = 15)
      .withColumn("terms", termsIdx2StrChineseWorst(col("termIndices")))

    val IndianRow1 = IndianTopics1.select("topic", "terms", "termWeights").collectAsList()
    output += "\n"+ IndianRow1.toArray.mkString(" ")

    val ChineseRow1 = ChineseTopics1.select("topic", "terms", "termWeights").collectAsList()
    output += "\n"+ ChineseRow1.toArray.mkString(" ")

    val sc = spark.sparkContext
    sc.parallelize(List(output)).saveAsTextFile(args(2))
    sc.stop()


  }
}
