from typing import Literal
from torch.utils.data import DataLoader
from elasticsearch import Elasticsearch, helpers
from elasticsearch.dsl import Search, Q, Query
from tqdm import tqdm
import pandas as pd
from fast.utils.env import Env


class ElasticFilter:
    def __init__(self, query: Query):
        self.query = query

    @staticmethod
    def range(field_name: str, start: float, end: float):
        query = {"range": {field_name: {"gte": start, "lte": end}}}

        return ElasticFilter(Q(query))

    @staticmethod
    def match(field_name: str, value: int):
        query = {"term": {field_name: value}}

        return ElasticFilter(Q(query))

    @staticmethod
    def str_match(field_name: str, value: str):
        query = {"match": {field_name: value}}

        return ElasticFilter(Q(query))

    @staticmethod
    def aggregate(filters: list["ElasticFilter"]):
        queries = []

        for f in filters:
            queries.append(f.query)

        query = Q("bool", must=queries)

        return ElasticFilter(query)


class Elastic:
    def __init__(self):
        env = Env()

        self.client = Elasticsearch(*env.elastic_args, **env.elastic_kwargs)

    def list_indices(self) -> dict:
        return self.client.options(ignore_status=[400, 404]).indices.get_alias(
            name="dataset"
        )

    def list_features(self, index: str) -> dict:
        mapping = self.client.indices.get_mapping(index=index)
        return mapping[index]["mappings"]["properties"]

    def calc_data_range(self, index: str) -> dict:
        fields = list(self.list_features(index).keys())

        s = Search(using=self.client, index=index)

        for field in fields:
            s.aggs.bucket(f"{field}_stats", "stats", field=field)

        res = s.execute()

        results = {
            field: {
                "min": res.aggregations[f"{field}_stats"].min,
                "max": res.aggregations[f"{field}_stats"].max,
            }
            for field in fields
        }

        return results

    def filter_data(self, index: str, filter: ElasticFilter):
        s = Search(using=self.client, index=index)
        s = s.query(filter.query)

        res = s.execute()

        hits = [hit.to_dict() for hit in res.hits]

        return hits

    def index_dataset(
        self,
        dataloader: DataLoader,
        feature_names: list[str],
        drop_index: bool = True,
        mode: Literal[
            "bulk",
            "streaming_bulk",
            "parallel_bulk",
        ] = "bulk",
    ):
        index = dataloader.dataset.__class__.__name__.lower()

        if drop_index:
            self.client.options(ignore_status=[400, 404]).indices.delete(index=index)
            self.client.options(ignore_status=[400, 404]).indices.delete_alias(
                index=index,
                name="dataset",
            )

        list_res = []

        for batch in tqdm(dataloader, desc="Indexing"):
            df = pd.DataFrame(batch, columns=feature_names)
            data = df.to_dict(orient="records")

            args = {
                "client": self.client,
                "actions": data,
                "index": index,
            }

            if mode == "bulk":
                res = helpers.bulk(**args)
            elif mode == "streaming_bulk":
                res = helpers.streaming_bulk(**args)
            elif mode == "parallel_bulk":
                res = helpers.parallel_bulk(**args)

            list_res.append(res)

        if index not in self.list_indices():
            self.client.indices.put_alias(index=index, name="dataset")

        return list_res
