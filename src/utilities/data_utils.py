from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, explode, split, when, lit, rand
from pyspark.sql.window import Window
from pymongo import MongoClient
import time
import logging



def fetch_data_from_mongo(spark: SparkSession, uri: str, db_name: str, collection_name: str):
    """
    Fetch data from a MongoDB collection into a PySpark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        A SparkSession object, already configured to use the Mongo Spark connector.
    uri : str
        The MongoDB connection URI (e.g. "mongodb://user:password@mongodb:27017").
    db_name : str
        The name of the MongoDB database.
    collection_name : str
        The name of the collection to read from.

    Returns
    -------
    pyspark.sql.DataFrame
        A Spark DataFrame containing the data from the specified MongoDB collection.
    """
    df = (spark.read
               .format("mongodb")
               .option("uri", uri)
               .option("database", db_name)
               .option("collection", collection_name)
               .load())
    return df

def preprocess_behaviors_mind(
    spark: SparkSession, 
    train_df: DataFrame,
    valid_df: DataFrame,
    npratio: int = 4
):
    logging.info(f"Starting to preprocess MIND dataset.")
    
    def process_behaviors(df):
        # Debugging: Print schema
        df.printSchema()
        
        # Select and rename columns
        df = df.select(
            col("impression_id").alias("impressionId"),
            col("user_id").alias("userId"),
            col("time").alias("timestamp"),
            col("history").alias("clicked"),
            col("impressions")
        )
        
        impressions_df = df.withColumn("impression", explode(split(col("impressions"), " ")))
        
        # Extract clicked status and newsId
        impressions_df = impressions_df.withColumn(
            "clicked",
            when(col("impression").endswith("-1"), lit(1)).otherwise(lit(0))
        ).withColumn(
            "newsId",
            split(col("impression"), "-")[0]
        ).select("userId", "newsId", "clicked")
        
        # Clean and cast userId and newsId
        impressions_df = impressions_df.withColumn("userId", F.regexp_replace(col("userId"), "^U", "").cast("int"))
        impressions_df = impressions_df.withColumn("newsId", F.regexp_replace(col("newsId"), "^N", "").cast("int"))
        
        impressions_df = impressions_df.dropna(subset=["userId", "newsId", "clicked"])
        
        # Positive and negative samples
        positive_samples = impressions_df.filter(col("clicked") == 1)
        negative_samples = impressions_df.filter(col("clicked") == 0).withColumn("rand", rand())
        
        # Select npratio negative samples per positive sample
        window = Window.partitionBy("userId").orderBy("rand")
        negative_samples = negative_samples.withColumn("rank", F.row_number().over(window)) \
                                           .filter(col("rank") <= npratio) \
                                           .drop("rank", "rand")
        
        combined_samples = positive_samples.union(negative_samples)
        
        return combined_samples

    # Apply preprocessing
    train_df = process_behaviors(train_df)
    valid_df = process_behaviors(valid_df)  # Corrected to process valid_df
    
    logging.info("Preprocessing of MIND dataset completed.")
    return train_df, valid_df


def wait_for_data(uri, db_name, collection_names, check_field, timeout=600, interval=10):
    """
    Poll the MongoDB collections to ensure data exists in each one.

    Parameters:
        uri (str): MongoDB URI.
        db_name (str): Database name.
        collection_names (list of str): List of collection names to check.
        check_field (str): Field to check for existence.
        timeout (int): Maximum time to wait for each collection in seconds.
        interval (int): Time between checks in seconds.
    """
    client = MongoClient(uri)
    db = client[db_name]

    # Iterate over each collection name provided
    for collection_name in collection_names:
        collection = db[collection_name]
        start_time = time.time()

        print(f"Checking collection '{collection_name}' for data...")

        # Wait for data to appear in the current collection
        while time.time() - start_time < timeout:
            if collection.find_one({check_field: {"$exists": True}}):
                print(f"Data found in collection '{collection_name}'.")
                break  # Move on to the next collection once data is found
            print(f"Waiting for data in collection '{collection_name}'...")
            time.sleep(interval)
        else:
            # Timeout reached without finding data in this collection
            raise TimeoutError(
                f"Data was not available in collection '{collection_name}' "
                f"within {timeout} seconds."
            )

    print("Data is available in all specified collections.")
    return True

def write_to_mongodb(df: DataFrame, MONGO_URI: str, DATABASE_NAME: str, COLLECTION_NAME: str):
    """
    Write the given DataFrame to the MongoDB embeddings collection efficiently.
    
    :param df: DataFrame to write
    :param MONGO_URI: MongoDB connection URI
    :param DATABASE_NAME: Target MongoDB database name
    :param COLLECTION_NAME: Target MongoDB collection name
    """
    (df.write
     .format("mongodb")
     .option("uri", MONGO_URI)
     .option("database", DATABASE_NAME)
     .option("collection", COLLECTION_NAME)
     .option("replaceDocument", "false")  # Append mode to avoid replacing existing documents
     .mode("append")
     .option("batchSize", 1000)  # Adjust based on MongoDB server capacity
     .option("w", "majority")     # Ensure write acknowledgment
     .save())
