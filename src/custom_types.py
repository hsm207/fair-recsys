from typing import Tuple, Callable, Dict, Any, Union, NamedTuple

import pyspark
from lenskit.algorithms import Algorithm

DateRange = Tuple[str, str]
DataFrame = pyspark.sql.DataFrame
ObjectiveFn = Callable[[Dict[str, Any]], Dict[str, Any]]
# a union of types in case you want to support learning algorithms from other libraries
Model = Union[Algorithm]

Movie = NamedTuple(
    "Movie", [("movie_id", str), ("base_score", float), ("category", str)]
)
