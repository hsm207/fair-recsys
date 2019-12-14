from hyperopt import STATUS_OK
from lenskit import util, batch, topn
from lenskit.algorithms import Recommender, als

from src.custom_types import *


# TODO: Figure out of Databricks's Hyperopt supports MLlib algos or not
# def create_objective_fn(
#     train_df: DataFrame, actual_ranks: DataFrame, recsize: int
# ) -> ObjectiveFn:
#     def objective_fn(params: Dict[str, Any]):
#         rank = params["rank"]
#         max_iter = params["maxIter"]
#         cold_start_strategy = "drop"  # no difference if switch to "nan"!
#         implicit_prefs = False
#
#         als = ALS(
#             rank=rank,
#             maxIter=max_iter,
#             coldStartStrategy=cold_start_strategy,
#             implicitPrefs=implicit_prefs,
#             userCol="userId",
#             itemCol="movieId",
#             ratingCol="adj_rating",
#         )
#
#         model = als.fit(train_df)
#
#         users = actual_ranks.select("userId").distinct()
#         predicted_ranks = (
#             model.recommendForUserSubset(users, recsize)
#             .rdd.map(lambda r: (r.userId, [row.movieId for row in r.recommendations]))
#             .toDF(["userId", "predicted_ranking"])
#         )
#
#         prediction_and_labels = (
#             predicted_ranks.join(actual_ranks, "userId", "inner").drop("userId").rdd
#         )
#
#         metrics = RankingMetrics(prediction_and_labels)
#
#         # maximize ndcg is the same as minimize -ndcg
#         loss = -metrics.ndcgAt(recsize)
#
#         return {"loss": loss, "status": STATUS_OK}
#
#     return objective_fn


def create_objective_fn(
    train_df: DataFrame, test_df: DataFrame, recsize: int
) -> ObjectiveFn:

    assert {"user", "item", "rating"}.issubset(train_df.columns)
    assert {"user", "item", "rating"}.issubset(test_df.columns)

    test_users = test_df.user.unique()

    def objective_fn(params: Dict[str, Any]):
        algo = als.BiasedMF(
            features=params["features"],
            iterations=params["iteration"],
            reg=0.1,
            damping=5,
        )

        model = util.clone(algo)
        model = Recommender.adapt(model)
        model.fit(train_df)

        recs = batch.recommend(model, test_users, recsize)

        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)

        results = rla.compute(recs, test_df)

        target_metric = -results.ndcg.mean()

        return {"loss": target_metric, "status": STATUS_OK}

    return objective_fn
