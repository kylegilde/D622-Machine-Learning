## DATA622 HW #4

- Assigned on October 16, 2018
- Due on October 30, 2018 12:00 PM EST
- 15 points possible, worth 15% of your final grade

### Instructions:

Use the two resources below to complete both the critical thinking and applied parts of this assignment.

1. Listen to all the lectures in Udacity's [Intro to Hadoop and Mapreduce](https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617) course.  

2. Read [Hadoop A Definitive Guide Edition 4]( http://javaarm.com/file/apache/Hadoop/books/Hadoop-The.Definitive.Guide_4.edition_a_Tom.White_April-2015.pdf), Part I Chapters 1 - 3.

### Critical Thinking (10 points total)

Submit your answers by modifying this README.md file.

1. (1 points) What is Hadoop 1's single point of failure and why is this critical?  How is this alleviated in Hadoop 2?

	* As Sarah Sproehnle from Udacity's [Intro to Hadoop and Mapreduce](https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617) explains in lesson 3.3, Hadoop's single point of failure (SPOF) used to be the namenode. Since the namenode contains the metadata for the cluster's data, if the namenode permanently failed, the cluster's data became inaccessible and effectively lost.

	* As Tom White explains in *Hadoop: the Definitive Guide* (pages 48-49), Hadoop 2 fixed this SPOF by "adding support for HDFS high availability." Practically speaking, Hadoop 2 has two instances of the namenode, an active one and standby one. The standby namenode takes periodic snapshots of the active namenode, and it can take over for the active one if it were to fail.

2. (2 points) What happens when a datanode fails?

 - Hadoop stores redundant copies of the data. According to Sproehnle in lesson 3.2, since each block of data is replicated on 3 different datanodes in the cluster, there are still 2 other copies of the block. If a datanode fails, the namenode will arrange to make another copy of the blocks. White writes that if a datanode fails, the namenode "notices that the block is under-replicated, and it arranges for a further replica to be created on another node" (73).

3. (1 point) What is a daemon?  Describe the role task trackers and job trackers play in the Hadoop environment. As 

In the Mapreduce process, Sproehnle explains that a daemon is just a piece of code running on a node. As Rohit Menon in the [Cloudera Administration Handbook](https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781783558964/1/ch01lvl1sec10/understanding-the-apache-hadoop-daemons) explains, Hadoop 1.x has 5 types of daemons:

	- namenode
	- secondary (standby) namenode
	- jobtracker
	- datanode
	- tasktracker


White explains that the jobtracker and tasktracker are the 2 types of daemons that contol the job execution process of Mapreduce 1 (83). 

The [hadoopinrealworld.com site](http://hadoopinrealworld.com/jobtracker-and-tasktracker/) conveys what these daemons do in more practical terms:

- A jobtracker receives the execution request from the client, finds the data on the namenode, selects a tasktracker near this data to execute the task, monitors the tasktrackers' progress and sends the overall job status back to the client. Per the Udacity course, the jobtracker tries to assign the tasks to a tasktracker that is already on the node that contains the data, but that is sometimes not possible if that tasktracker is already busy.

- The tasktrackers run on datanodes, and they execute the mapper and reducer tasks. They communicate the task progress to the jobtracker. If a tasktracker fails, the jobtracker will assign the task to another node.

In Mapreduce 2, the jobtracker and tasktracker have been replaced by the ResourceManager and Node Manager, respectively.

4. (1 point) Why is Cloudera's VM considered pseudo-distributed computing?  How is it different from a true Hadoop cluster computing?

	* It is consider pseudo-distributed computing because the VM has the entire cluster on a single node. The namenode and datanodes are all running on a single machine. All data would be lost if the node failed, because this configuration does not have redundant copies of the data on other machines. 

5. (1 point) What is Hadoop streaming? What is the Hadoop Ecosystem?

	* As the Udacity course explains, even though mapreduce is written Java, Hadoop streaming allows users to use Python or other languages

	* White describes Hadoop streaming as an interface between Hadoop and your Mapreduce program that uses Unix standard streams. One can use any computer language that reads standard input and writes standard output (37).

6. (1 point) During a reducer job, why do we need to know the current key, current value, previous key, and cumulative value, but NOT the previous value?

	* The reducer job needs to know the previous and current keys so that it knows when keys change, which indicates that it should initiate a new cumulative value for the new key. 

	* The reducer jobs needs to know the cumulative value and current value so that it can add the current value to the cumulative value while processing one key.

	* It does not need to know the previous value because it has already been accounted for when it was added to the cumulative value.

7. (3 points) A large international company wants to use Hadoop MapReduce to calculate the **# of sales by location by day.**  The logs data has one entry per location per day per sale.  Describe how MapReduce will work in this scenario, using key words like: intermediate records, shuffle and sort, mappers, reducers, sort, key/value, task tracker, job tracker.  

	* When a Mapreduce job is executed, the jobtracker daemon uses the metadata on the namenode to find the datanodes containing the blocks of data that have logs for the requested locations and dates. The jobtracker then assigns the Mapreduce tasks to tasktrackers on each of the required datanodes. 

	* The tasktrackers execute the mapper and reducer tasks on the blocks of data in the datanodes. If a tasktracker fails, the jobtracker will assign the task to another node & tasktracker.

	* At the tasktrackers' direction, the mappers process the sales logs in parallel. They parse the delimited data of each log, and the requested attributes are stored in a key-value pair known as the intermediate records. Since the company has effectively given us 2 keys in our example, the mapper would need to create a location-date key using an array-type object, e.g. a Python tuple. It could also concatenate the location and date with a delimiter into a single string. Since we are interested in the count of sales, the value would be assigned as 1. These key-value pairs are the output of the mapper.

	* Next, the intermediate records are shuffled and sorted. The intermediate records are the input to the reducers. The *shuffle* is the transfer of the intermediate records from the mappers to the reducers. The reducers *sort* our location-date keys in order and then process the values. Since we are counting sales, the reducer would sum all of the values of 1. Finally, the output is the unique location-date keys and the count of the sales logs. This is written to HDFS.

### Applied (5 points total)

Submit the mapper.py and reducer.py and the output file (.csv or .txt) for the first question in lesson 6 for Udacity.  (The one labelled "Quiz: Sales per Category")  Instructions for how to get set up is inside the Udacity lectures.  

The value of total sales:

- Toys: 57,463,477.11

- Consumer Electronics: 57,452,374.13

Commands:

* `hadoop fs -put purchases.txt myinput`
* `hs mapper.py reducer.py myinput myoutput`
* `hadoop fs -cat myoutput/part-00000 | less`
* `hadoop fs -cat myoutput/part-00000 | tee myoutput.txt`
