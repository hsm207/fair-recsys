import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    IntegerType,
    FloatType,
    TimestampType,
)

from src.custom_types import *

schema = StructType(
    [
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", FloatType(), False),
        StructField("timestamp", DoubleType(), False),
    ]
)


def get_data(ratings_path: str) -> DataFrame:
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv(
        ratings_path, encoding="utf-8", schema=schema, header=True
    ).withColumn("timestamp", f.col("timestamp").cast(TimestampType()))

    return df


def add_label(df: DataFrame) -> DataFrame:
    return df.withColumn("adj_rating", f.expr("rating - 2.5"))


def timesplit_data(
    df: DataFrame,
    train_dates: DateRange = ("2014-01-01", "2014-11-01"),
    valid_dates: DateRange = ("2014-11-01", "2014-12-01"),
    test_dates: DateRange = ("2014-12-01", "2015-01-01"),
) -> Tuple[DataFrame, DataFrame, DataFrame]:

    date_ranges = [train_dates, valid_dates, test_dates]
    df_splits = tuple(
        df.filter(f.expr(f"timestamp BETWEEN '{start_date}' AND '{end_date}'"))
        for start_date, end_date in date_ranges
    )

    return df_splits


def create_ground_truth_rankings(df: DataFrame) -> DataFrame:
    assert {"userId", "rating", "movieId"}.issubset(df.columns)

    return (
        df.withColumn("is_relevant", f.expr("if(rating > 3.5, true, false)"))
        .filter(f.col("is_relevant"))
        # for each user, order the relevant movies by highest rated first
        .withColumn("rmp", f.struct(f.col("rating"), f.col("movieId")))
        .groupBy("userId")
        .agg(f.sort_array(f.collect_list("rmp"), asc=False).alias("rating_movie_pairs"))
        .rdd.map(lambda r: (r.userId, [row.movieId for row in r.rating_movie_pairs]))
        .toDF(["userId", "actual_ranking"])
    )
