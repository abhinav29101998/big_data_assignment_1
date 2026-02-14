# PySpark Assignment 1



##  Overview

This repository contains the implementation and analysis for ** Assignment 1**, which focuses on distributed data processing using:

* **Apache Hadoop (MapReduce)**
* **Apache Spark (PySpark)**

The assignment demonstrates large-scale data processing, text analytics, document similarity, and graph-based author influence modeling using the Project Gutenberg dataset.



##  Objectives

1. Install and configure a **single-node Hadoop cluster**
2. Execute the **WordCount MapReduce program**
3. Process large text datasets using **Apache Spark**
4. Extract metadata from books
5. Compute **TF-IDF scores**
6. Measure **cosine similarity between books**
7. Build an **Author Influence Network**



##  Technologies Used

* Ubuntu (WSL on Windows)
* Java JDK 8
* Apache Hadoop 3.x
* Apache Spark 3.x
* PySpark
* Python
* Project Gutenberg Dataset

##  PART 1 — Hadoop MapReduce

###  Hadoop Setup

* Installed Hadoop in Ubuntu (WSL)
* Configured HDFS
* Started NameNode and DataNode
* Verified using `jps`

###  WordCount Execution

Steps:

```bash
# Upload file to HDFS
hdfs dfs -copyFromLocal input.txt /input

# Run MapReduce job
hadoop jar hadoop-mapreduce-examples.jar wordcount /input /output

# View output
hdfs dfs -cat /output/part-r-00000
```

Output shows word frequency counts.

 

##  PART 2 — Apache Spark Processing

###  Dataset Loading

All Gutenberg books are loaded into a Spark DataFrame.

Columns:

* file_name
* text

 

###  Metadata Extraction

Extracted using regex:

* Title
* Release Date
* Language
* Encoding

Analysis performed:

* Books per year
* Most common language
* Average title length

 

###  TF-IDF Computation

Processing steps:

1. Lowercase conversion
2. Remove punctuation
3. Tokenization
4. Stopword removal
5. Term Frequency calculation
6. Inverse Document Frequency
7. TF-IDF vector generation

TF-IDF identifies important words in each book.

 

###  Cosine Similarity

Each book represented as a TF-IDF vector.

Similarity formula:

```
similarity = (A · B) / (|A| × |B|)
```

Used to identify most similar books.

 

##  PART 3 — Author Influence Network

Authors connected if their books were published within a defined time window (e.g., 5 years).

Graph analysis:

* In-degree
* Out-degree
* Top influential authors

 

## Scalability Considerations

Pairwise similarity grows quadratically with number of documents.

Spark improves performance using:

* Distributed processing
* Parallel execution
* In-memory computation

 

##  How to Run

### Start Hadoop

```bash
start-dfs.sh
```

### Run WordCount

```bash
hadoop jar WordCount.jar /input /output
```

### Run Spark Programs

```bash
spark-submit load_books.py
spark-submit metadata.py
spark-submit tfidf.py
spark-submit similarity.py
spark-submit author_influence.py
```

 

 

