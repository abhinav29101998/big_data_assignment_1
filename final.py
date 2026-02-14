from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, abs

# Initialize Spark session
spark = SparkSession.builder.appName("Q12_AuthorInfluenceGraph").getOrCreate()

# ---------------------------------------------------
# Step 1: Read files and capture Author + Year info
# ---------------------------------------------------
files_rdd = spark.sparkContext.wholeTextFiles("books_data/*.txt")
books_df = files_rdd.toDF(["file_path", "content"])

author_year_df = (
    books_df
    .withColumn("author_name", regexp_extract(col("content"), r"Author:\s*(.*)", 1))
    .withColumn("publish_year", regexp_extract(col("content"), r"Release [Dd]ate:.*(\d{4})", 1).cast("int"))
    .filter("author_name != '' AND publish_year IS NOT NULL")
    .select("author_name", "publish_year")
    .distinct()
)

# ---------------------------------------------------
# Step 2: Build influence network (5-year window)
# ---------------------------------------------------
left_authors = author_year_df.alias("left")
right_authors = author_year_df.alias("right")

# Recreate dataset using file name as author identifier
author_year_df = (
    books_df
    .withColumn("author_name", regexp_extract(col("file_path"), r"([^/]+)$", 1))
    .withColumn("publish_year", regexp_extract(col("content"), r"Release [Dd]ate:.*(\d{4})", 1).cast("int"))
    .filter("publish_year IS NOT NULL")
    .select("author_name", "publish_year")
    .distinct()
)

connections_df = (
    left_authors.join(
        right_authors,
        (col("left.author_name") != col("right.author_name")) &
        (abs(col("left.publish_year") - col("right.publish_year")) <= 5)
    )
    .select(
        col("left.author_name").alias("influenced_by"),
        col("right.author_name").alias("author")
    )
)

# ---------------------------------------------------
# Step 3: Compute influence metrics (degrees)
# ---------------------------------------------------
incoming_links = (
    connections_df
    .groupBy("author")
    .count()
    .withColumnRenamed("count", "in_degree")
)

outgoing_links = (
    connections_df
    .groupBy("influenced_by")
    .count()
    .withColumnRenamed("count", "out_degree")
)

print("=== Most Influenced Authors (Top 5 by In-Degree) ===")
incoming_links.orderBy(col("in_degree").desc()).show(5)

spark.stop()
