## Fall 2018 DATA622.001 Homework #5
- Assigned on October 30, 2018
- Due on November 13, 2018 12:00 PM EST
- 15 points possible, worth 15% of your final grade

### Instructions:

Read the following:
- [Apache Spark Python 101](https://www.datacamp.com/community/tutorials/apache-spark-python)
- [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)

Optional Readings:
- [Paper on RDD](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)
- [Advanced Analytics with Spark: Patterns for Learning from Data at Scale, 2nd Edition](https://www.amazon.com/_/dp/1491972955), Chapters 1 - 2

Additional Resources:
- [Good intro resource on PySpark](https://annefou.github.io/pyspark/slides/spark/#1)
- [Spark The Definitive Guide](https://github.com/databricks/Spark-The-Definitive-Guide)
- [Google Cloud Dataproc, Spark on GCP](https://codelabs.developers.google.com/codelabs/cloud-dataproc-starter/)


### Critical Thinking (8 points total)

#### 1. How is the current Spark's framework different from MapReduce?  What are the tradeoffs of better performance speed in Spark?

Hadoop and Spark are both open-source, big-data frameworks, but they differ in their specific functionalities. As the [educational site Educba](https://www.educba.com/mapreduce-vs-apache-spark/) explains, Hadoop is composed of two layers. The Hadoop distributed file system (HDFS) is used for data storage, and Mapreduce is used for data processing. On the other hand, while lacking data storage, Spark is an "independent data processing engine," and it can be integrated with HDFS, other distributed file systems, Google Cloud Platform or AWS.

Both Mapreduce and Spark are used to process distributed data, but they have different advantages. While Hadoop is known for its scalability, Spark is known for its performance speed. Mapreduce can only process data on disk while Spark can process data 10 times faster on disk and 100 times faster in memory. The reason that Hadoop Mapreduce is so slow is that it is constantly reading and writing to disk. Every interim result of each process is written to disk.

Mapreduce can only do batch processing and has a high amount of latency. However, Spark can do the real-time data processing required with online transaction processing (OLTP). Spark's low latency & in-memory caching make it the ideal tool for iterative and interactive analysis, but the use of this RAM makes Spark more costly than Mapreduce.

Hadoop Mapreduce is written in Java, and Spark is written in Scala. While Mapreduce supports the programming languages of C, C++, Ruby, Groovy, Perl & Python, Spark supports Java, Python & R. Both have their own SQL implementations. 

#### 2. Explain the difference between Spark RDD and Spark DataFrame/Datasets.

As Karlijin Willems explains in Datacamp's [Apache Spark in Python: Beginner's Guide](https://www.datacamp.com/community/tutorials/apache-spark-python), the resilent distributed datasets (RDDs) are the "building blocks of Spark." They are sets of Java or Scala objects that represent data. Anne Fouilloux, in her [Big Data slides](https://annefou.github.io/pyspark/slides/spark/#12), explains that RDDs are considered resilent because they are fault-tolerant, which means that Spark will rebuild the data if there is a failure during processing.

From the lecture, we learned that RDDs are immutable, which means that every change to the data yields a new RDD. The user can go back to all of the previous versions of the RDD. Willems also describes how RDDs are also compile-time type safe, which means that the data types are validated when the code is compiled and before the operations are executed. She writes that one disadvantage of RDDs is that they can "build inefficient transformation chains." When we are using non-Java virtual machine languages like Python, Spark cannot optimize these chains of operations.

Willems explains that because of the RDD limitations, the DataFrame API was created. This API is a higher level abstraction that allows the user to manipulate the data. Practically speaking, this means that DataFrames are given the logical plan but Spark is allowed to create the most efficient physical plan. It is this optimized execution plan along with the DataFrame's custom memory management that make DataFrames faster than RDDs.

As Willems explains, the only disadvantage between the RDD and DataFrame is that the DataFrame is not type safe. Consequently, the [Dataset](https://spark.apache.org/docs/2.3.0/api/java/index.html?org/apache/spark/sql/Dataset.html) offers a strongly typed version of the DataFrame.

For performance reasons, she recommends that users avoid passing data between RDDs and DataFrames because the serialization and deserialization are expensive tasks.

#### 3. Explain the difference between SparkML and Mahout.  

As Andrew Oliver in [Forgot about Mahout? It’s back, and worth your attention](https://www.infoworld.com/article/3197429/machine-learning/forgot-about-mahout-its-back-and-worth-your-attention.html), Mahout was originally launched as machine learning framework for Hadoop using Mapreduce. With the rise of the faster data processing led by Spark, the development for the Mapreduce version of Mahout was frozen in 2014, and the Scala version of Mahout began. This new version is "engine-neutral," but Spark is recommended.

However, the old and new versions of the Mahout frameworks still have not reached feature parity. From the site's [Algorithm menu](https://mahout.apache.org/docs/latest/algorithms/recommenders/), we can see that the Scala version supports some linear algebra, preprocessing, regression & recommenders. However, it is still missing classification algorithms. The [Mapreduce-version features](https://mahout.apache.org/docs/latest/algorithms/map-reduce/) include several classifers, ranging from naive Bayes to neural networks, but it lacks the linear algebra, preprocessing & recommender functionality.

Spark's machine learning framework consists of two packages. After a close inspection of Mahout and [Spark's ML features](https://spark.apache.org/docs/latest/ml-pipeline.html), Spark appears to have nearly all of Mahout's features between its two libraries, and it has many more feature extraction, transformation & selection functions.

#### 4. Explain the difference between Spark.mllib and Spark.ml.

Spark has two machine learning libraries/APIs. Karau and Warren in [*High Performance Spark*](https://books.google.com/books?id=90glDwAAQBAJ&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false) write that MLlib came first, but as of 2017, it entered a maintenance-only mode. The newer spark.ml (ML) is in active development, but it does not have all of MLlib's functionality yet. These two libraries utilize Spark's different data types. MLlib supports the RDDs while ML uses DataFrames and Datasets. 

The two libraries have different design philosophies. MLlib offers only the machine learning algorithms. In scikit-learn fashion, ML offers more than just the algorithms and includes data pipelining, data-cleaning and feature-selection functionality. 

After comparing the [list of features](https://spark.apache.org/docs/latest/ml-statistics.html) for each package, ML offers many more feature extraction, transformation & selection functions as well as more classification & regression algorithms. Currently, only MLlib supports streaming data. 

If the user does not need streaming data or MLlib-specfic functionality, then ML is likely the best library to choose since it is fully supported and in active feature development.

#### 5. Explain the tradeoffs between using Scala vs PySpark.

In terms of performance & concurrency, Willems writes that Spark's native language of Scala is superior to PySpark. Scala is better for larger production projects. It supports asynchronous code, which allows for non-blocking I/O calls. This means that I/O operations can be executed in parallel, and this is highly advantageous when working with streaming data. Additionally, Scala offers type safety that Python does not.

On the other hand, PySpark is better for smaller projects and data science. Python's verbosity makes it easier to learn and more readable than Scala. PySpark can also be used with Python's abundance of machine learning and natural language processing tools.

### Applied (7 points total)

Submit your Jupyter Notebook from following along with [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)
