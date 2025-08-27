import os
import sys
import uuid
from google.cloud import aiplatform

try:
    PROJECT_ID = os.environ["PROJECT_ID"]
    REGION = os.environ["REGION"]
    BUCKET_URI = os.environ["BUCKET_URI"]
    SERVICE_ACCOUNT = os.environ["SERVICE_ACCOUNT"]
    HF_TOKEN = os.environ["HF_TOKEN"]
    BASE_MODEL_ID = os.environ["BASE_MODEL_ID"]
    LORA_ADAPTER_GCS_PATH = os.environ["LORA_ADAPTER_GCS_PATH"]
    BASE_MODEL_GCS_PATH = f"{BUCKET_URI}/models/deepseek-r1-70b-local"
except KeyError as e:
    print(f"Error: Missing required environment variable: {e}")
    sys.exit(1)

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

machine_type = "a3-highgpu-4g"
accelerator_type = "NVIDIA_H100_80GB"
accelerator_count = 4
serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250506_0916_RC01"

vllm_args = [
    "python", "-m", "vllm.entrypoints.api_server",
    "--host=0.0.0.0",
    "--port=8080",
    f"--model={BASE_MODEL_GCS_PATH}",
    f"--tensor-parallel-size={accelerator_count}",
    "--swap-space=16",
    "--gpu-memory-utilization=0.85",
    "--max-model-len=8192",
    "--disable-log-stats",
    "--enable-lora",
    "--max-loras=1",
]

env_vars = {
    "MODEL_ID": BASE_MODEL_ID,
    "HF_TOKEN": HF_TOKEN,
}

model = aiplatform.Model.upload(
    display_name=f"{BASE_MODEL_ID.split('/')[-1]}-{uuid.uuid4().hex[:4]}",
    serving_container_image_uri=serving_container_image_uri,
    serving_container_args=vllm_args,
    serving_container_ports=[8080],
    serving_container_predict_route="/generate",
    serving_container_health_route="/ping",
    serving_container_environment_variables=env_vars,
)

endpoint = model.deploy(
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    service_account=SERVICE_ACCOUNT,
)

instance = {
    "prompt": "A patient presents with fever and a rash. What are your top 3 differential diagnoses?<|eot_id|>",
    "max_tokens": 1024,
    "temperature": 0.2,
    "lora_request": {
        "lora_adapter_name": "heidi-adapter",
        "lora_adapter_id": LORA_ADAPTER_GCS_PATH,
    }
}

response = endpoint.predict(instances=[instance])

print(response.predictions[0])
