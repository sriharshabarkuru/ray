import requests
from starlette.requests import Request
from typing import Dict

from transformers import pipeline
import ray
from ray import serve
import time
from ray.serve.deployment_graph import InputNode
from ray.serve.drivers import DAGDriver


# 1: Wrap the pretrained sentiment analysis model in a Serve deployment.
@serve.deployment(route_prefix="/sentiment")
class SentimentAnalysisDeployment:
    def __init__(self):
        print("init ...")
        self._model = pipeline("sentiment-analysis")

    def __call__(self, request: Request) -> Dict:
        print("query params", request.query_params["text"])
        return self._model(request.query_params["text"])[0]


# 2: Deploy the deployment.
# serve.run(SentimentAnalysisDeployment.bind())

with InputNode() as msg:
    deployment_graph = SentimentAnalysisDeployment.bind()
