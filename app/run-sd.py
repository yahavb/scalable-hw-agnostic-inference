import os
import boto3
import math
import time
import random
import gradio as gr
from matplotlib import image as mpimg
from fastapi import FastAPI
import torch
from typing import Optional
from PIL import Image
import base64
from io import BytesIO

app_name=os.environ['APP']
pod_name=os.environ['POD_NAME']
nodepool=os.environ['NODEPOOL']
model_id=os.environ['MODEL_ID']
device=os.environ["DEVICE"]
compiled_model_id=os.environ['COMPILED_MODEL_ID']
num_inference_steps=int(os.environ['NUM_OF_RUNS_INF'])
cw_namespace='hw-agnostic-infer'
cloudwatch = boto3.client('cloudwatch', region_name='us-west-2')

def cw_pub_metric(metric_name,metric_value,metric_unit):
  response = cloudwatch.put_metric_data(
    Namespace=cw_namespace,
    MetricData=[
      {
        'MetricName':metric_name,
        'Value':metric_value,
        'Unit':metric_unit,
       },
    ]
  )
  print(f"in pub_deployment_counter - response:{response}")
  return response
# Define datatype
DTYPE = torch.bfloat16

if device=='xla':
  from optimum.neuron import NeuronStableDiffusionPipeline 
elif device=='cuda' or device=='triton':
  from diffusers import StableDiffusionPipeline

from diffusers import DDIMScheduler


def benchmark(n_runs, test_name, model, model_inputs):
    if not isinstance(model_inputs, tuple):
        model_inputs = model_inputs

    #warmup_run = model(**model_inputs)

    latency_collector = LatencyCollector()

    for _ in range(n_runs):
        latency_collector.pre_hook()
        print(model_inputs)
        res = model(**model_inputs)
        latency_collector.hook()

    p0_latency_ms = latency_collector.percentile(0) * 1000
    p50_latency_ms = latency_collector.percentile(50) * 1000
    p90_latency_ms = latency_collector.percentile(90) * 1000
    p95_latency_ms = latency_collector.percentile(95) * 1000
    p99_latency_ms = latency_collector.percentile(99) * 1000
    p100_latency_ms = latency_collector.percentile(100) * 1000

    report_dict = dict()
    report_dict["Latency P0"] = f'{p0_latency_ms:.1f}'
    report_dict["Latency P50"]=f'{p50_latency_ms:.1f}'
    report_dict["Latency P90"]=f'{p90_latency_ms:.1f}'
    report_dict["Latency P95"]=f'{p95_latency_ms:.1f}'
    report_dict["Latency P99"]=f'{p99_latency_ms:.1f}'
    report_dict["Latency P100"]=f'{p100_latency_ms:.1f}'

    report = f'RESULT FOR {test_name} on {pod_name}:'
    for key, value in report_dict.items():
        report += f' {key}={value}'
    print(report)
    return report

class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

if device=='xla':
  pipe = NeuronStableDiffusionPipeline.from_pretrained(compiled_model_id)
elif device=='cuda' or device=='triton':
  pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,torch_dtype=DTYPE).to("cuda")
  pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  if device=='triton':
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.unet = torch.compile(
      pipe.unet, 
      fullgraph=True, 
      mode="max-autotune-no-cudagraphs"
    )

    pipe.text_encoder = torch.compile(
      pipe.text_encoder,
      fullgraph=True,
      mode="max-autotune-no-cudagraphs",
    )

    pipe.vae.decoder = torch.compile(
      pipe.vae.decoder,
      fullgraph=True,
      mode="max-autotune-no-cudagraphs",
    )

    pipe.vae.post_quant_conv = torch.compile(
      pipe.vae.post_quant_conv,
      fullgraph=True,
      mode="max-autotune-no-cudagraphs",
    )
  pipe.enable_attention_slicing()

def text2img(prompt):
  start_time = time.time()
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  image = pipe(**model_args).images[0]
  total_time =  time.time()-start_time
  return image, str(total_time)

prompt="portrait photo of a old warrior chief"
model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
image = pipe(**model_args).images[0]

app = FastAPI()
io = gr.Interface(fn=text2img,inputs=["text"],
    outputs = [gr.Image(height=512, width=512), "text"],
    title = model_id + ' in AWS EC2 ' + device + ' instance; pod name ' + pod_name + ';portrait photo of a cat ,detailed,8k')

@app.get("/")
def read_main():
  return {"message": "This is" + model_id + " pod " + pod_name + " in AWS EC2 " + device + " instance; try /load/{n_runs}/infer/{n_inf}; /genimage http post with user prompt "}

@app.get("/load/{n_runs}/infer/{n_inf}")
def load(n_runs: int,n_inf: int):
  start_time = time.time()
  prompt = "a photo of an astronaut riding a horse on mars"
  num_inference_steps = n_inf
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  report=benchmark(n_runs, "stable_diffusion_512", pipe, model_args)
  total_time =  time.time()-start_time

  counter_metric=app_name+'-counter'
  cw_pub_metric(counter_metric,1,'Count')
  
  counter_metric=nodepool
  cw_pub_metric(counter_metric,1,'Count')

  latency_metric=app_name+'-latency'
  cw_pub_metric(latency_metric,total_time,'Seconds')

  return {"message": "benchmark report:"+report}

def serialize_image(image: Image.Image) -> str:
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_str


@app.post("/genimage")
def generate_image_post(request: dict):
  prompt = request.get("prompt")
  response_image, latency = text2img(prompt)
  response_data = {
        "prompt": prompt,
        "response": serialize_image(response_image),
        "latency": latency
  }
  return response_data

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

app = gr.mount_gradio_app(app, io, path="/serve")
