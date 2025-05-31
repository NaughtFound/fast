import os
from dotenv import load_dotenv


class Env:
    def __init__(self) -> None:
        self.elastic_args = ()
        self.elastic_kwargs = {"hosts": os.environ.get("ELASTIC_HOSTS")}

    @staticmethod
    def load() -> None:
        load_dotenv()
