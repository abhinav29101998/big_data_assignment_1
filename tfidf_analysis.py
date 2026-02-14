from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, explode, count, log
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# ---------------------------------------------------
# Step 1: Create Spark session
# ---------------------------------------------------
spark = SparkSession.builder.appName("Q11_TextTFIDFAnalysis").getOrCreate()

# ---------------------------------------------------
# Step 2: Load text files from books_data directory
# DataFrame columns -> (path, content)
# ---------------------------------------------------
files_rdd = spark.sparkContext.wholeTextFiles("books_data/*.txt")
documents_df = files_rdd.toDF(["path", "content"])

# Extract simple file name from path
documents_df = documents_df.withColumn(
    "doc_name",
    regexp_replace(col("path"), ".*/", "")
)

# ---------------------------------------------------
# Step 3: Text cleaning and token preparation
# ---------------------------------------------------
normalized_df = documents_df.withColumn("content", lower(col("content")))
normalized_df = normalized_df.withColumn(
    "content",
    regexp_replace(col("content"), "[^a-z\\s]", "")
)

# Convert text into tokens (word list)
tokenizer = Tokenizer(inputCol="content", outputCol="tokens_raw")
tokenized_df = tokenizer.transform(normalized_df)

# Remove common stop words
stopword_filter = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens")
prepared_df = stopword_filter.transform(tokenized_df).select("doc_name", "tokens")

# ---------------------------------------------------
# Step 4: TF-IDF computation
# ---------------------------------------------------
# Expand token arrays into individual rows
exploded_tokens = prepared_df.withColumn(
    "term",
    explode(col("tokens"))
).filter(col("term") != "")

# Term Frequency (TF)
term_frequency = exploded_tokens.groupBy("doc_name", "term") \
                                .agg(count("term").alias("term_freq"))

# Inverse Document Frequency (IDF)
total_documents = documents_df.count()

doc_frequency = exploded_tokens.select("term", "doc_name") \
                               .distinct() \
                               .groupBy("term") \
                               .agg(count("doc_name").alias("docs_with_term"))

idf_values = doc_frequency.withColumn(
    "idf_value",
    log(total_documents / col("docs_with_term"))
)

# Combine TF and IDF
tfidf_df = term_frequency.join(idf_values, "term") \
                         .withColumn("tfidf", col("term_freq") * col("idf_value"))

print("=== Top 10 Highest TF-IDF Terms ===")
tfidf_df.orderBy(col("tfidf").desc()).show(10)

# ---------------------------------------------------
# Step 5: Identify common themes across books
# ---------------------------------------------------
print("=== Shared Important Terms Across Documents ===")
term_sharing = tfidf_df.groupBy("term") \
                       .agg(count("doc_name").alias("appears_in"))

frequent_shared_terms = term_sharing.filter(col("appears_in") > 1) \
                                    .orderBy(col("appears_in").desc())

frequent_shared_terms.show(5)

spark.stop()
