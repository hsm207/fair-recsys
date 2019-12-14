# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook downloads the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Steps

# COMMAND ----------

# MAGIC %md
# MAGIC Download and unzip the data:

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -q http://files.grouplens.org/datasets/movielens/ml-20m.zip
# MAGIC unzip -q ml-20m.zip
# MAGIC ls -la

# COMMAND ----------

# MAGIC %md
# MAGIC Move the data to dbfs:

# COMMAND ----------

# MAGIC %sh
# MAGIC mv  ./ml-20m /dbfs/tmp
# MAGIC ls -la /dbfs/tmp/ml-20m
