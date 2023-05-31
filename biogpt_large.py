#https://huggingface.co/microsoft/BioGPT-Large
from transformers import pipeline
import torch
import json
import os

from ray import serve

from transformers import BioGptTokenizer, BioGptForCausalLM

#@serve.deployment(route_prefix="/biogpt-large", ray_actor_options={"num_gpus": 1})
@serve.deployment(route_prefix="/biogpt-large", health_check_timeout_s=600)
class BioGptLarge:
    def __init__(self):
        self.pipe_biogpt = pipeline("text-generation", model=os.environ["MODEL_PATH"])


    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        data = json.loads(request)
        prompt = data['prompt']
        max_length = data.get('max_length', 100)
        num_sequences = data.get("num_sequences", 5)
        output_biogpt = self.pipe_biogpt(prompt, max_length=max_length, num_return_sequences=num_sequences, do_sample=True)
        result = output_biogpt[0]["generated_text"]
        return result

deploy = BioGptLarge.bind()
