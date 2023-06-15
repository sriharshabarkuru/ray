import logging
from typing import Optional

import mlflow.pyfunc
import pandas as pd
import ray
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from ray import serve
from starlette.requests import Request
import os

try:
    import ujson as json
except ModuleNotFoundError:
    import json

logger = logging.getLogger(__name__)


@serve.deployment()
class MLflowDeployment:
    def __init__(self):
        print("model path", os.environ["MODEL_PATH"])
        # self.model = mlflow.pyfunc.load_model(model_uri=model_uri)
        self.model = mlflow.pyfunc.load_model(model_uri=os.environ["MODEL_PATH"])

    async def predict(self, df):
        return self.model.predict(df).to_json(orient="records")

    async def _process_request_data(self, request: Request) -> pd.DataFrame:
        body = await request.body()
        if isinstance(body, pd.DataFrame):
            return body
        return pd.read_json(json.loads(body))

    async def __call__(self, request: Request):
        df = await self._process_request_data(request)
        return self.model.predict(df).to_json(orient="records")


# 2: Deploy the deployment.
deploy = MLflowDeployment.bind()
