from typing import Literal
from torch.utils.data import DataLoader
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import pandas as pd
from utils.env import Env


class Elastic:
    def __init__(self):
        env = Env()

        self.client = Elasticsearch(*env.elastic_args, **env.elastic_kwargs)

    def list_indices(self):
        return self.client.options(ignore_status=[400, 404]).indices.get_alias(
            name="dataset"
        )

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
