# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook creates an evaluation dataset to test the efficacy of the personalized reranking. Best run on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC # Libraries

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

from lenskit import batch, topn, util
from lenskit.algorithms import Recommender, als
from lenskit import topn
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

distant_tail_movies = spark.read.parquet("/tmp/ml-20m/movie_categories.parquet") \
  .filter(F.expr("category = 'distanttail'")) \
  .withColumnRenamed("movieId", "item")

# COMMAND ----------

train_df = spark.read.parquet("/tmp/ml-20m/train_df.parquet/") \
  .join(distant_tail_movies, "item", "left_anti") \
  .toPandas()

test_df = spark.read.parquet("/tmp/ml-20m/test_df.parquet/") \
  .join(distant_tail_movies, "item", "left_anti") \
  .toPandas()

# COMMAND ----------

user_pref = spark.read.parquet("/tmp/ml-20m/user_preference.parquet")

# COMMAND ----------

algo = als.BiasedMF(
            features=382,
            iterations=1,
            reg=0.1,
            damping=5,
        )

model = util.clone(algo)
model = Recommender.adapt(model)
model.fit(train_df)

# COMMAND ----------

# have a subset of test users with equal distribution between longtail and shorthead preference
df1 = test_df.merge(user_pref.toPandas(), "left", left_on="user", right_on="userId")
df2 = df1.query("longtail_pref >= 0.5").sample(n=250, random_state=123)
df3 = df1.query("longtail_pref < 0.5").sample(n=250, random_state=123)

test_users = pd.concat([df2, df3]).user.unique()

recs = batch.recommend(model, test_users, 100)

rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)

results = rla.compute(recs, test_df)

print(f"NDCG: {results.ndcg.mean()}")

# COMMAND ----------

# MAGIC %md
# MAGIC Save the results of `recs`:

# COMMAND ----------

recs.to_parquet("/dbfs/tmp/ml-20m/evaluation_dataset.parquet")
