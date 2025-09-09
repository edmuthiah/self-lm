import os
from dotenv import load_dotenv
import utils
from typing import Tuple
from google.cloud import aiplatform

load_dotenv()

project_id =  os.getenv('PROJECT_ID')
region = os.getenv('REGION')
base_model_gcs=os.getenv('BASE_MODEL_BUCKET_PATH')
lora_adapter_gcs=os.getenv('LORA_ADAPTER_BUCKET_PATH')
service_account=os.getenv('SERVICE_ACCOUNT')


machine_type='g2-standard-16'
accelerator_type='NVIDIA_L4'
serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250506_0916_RC01"

endpoint_path= 'projects/PROJECT_NUMBER/locations/us-central1/endpoints/ENDPOINT_ID'

def main():

    base_model_name = 'DeepSeek-R1-Distill-Llama-8B'
    model_id = base_model_name

    models["vllm_gpu"], endpoints["vllm_gpu"] = deploy_model_vllm(
        model_name=utils.get_job_name_with_datetime(prefix=base_model_name),
        model_id=model_id,
        base_model_id=model_id,
        base_model_bucket_path=base_model_gcs,
        service_account=service_account,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,
        enforce_eager=True,
        enable_lora=True,
        endpoint_path=endpoint_path
    )

    print(models)

def deploy_model_vllm(
    model_name: str,
    model_id: str,
    service_account: str,
    base_model_id: str = None,
    base_model_bucket_path: str = None,
    machine_type: str = "g2-standard-8",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto",
    enable_trust_remote_code: bool = False,
    enforce_eager: bool = False,
    enable_lora: bool = False,
    enable_chunked_prefill: bool = False,
    enable_prefix_cache: bool = False,
    host_prefix_kv_cache_utilization_target: float = 0.0,
    max_loras: int = 1,
    max_cpu_loras: int = 8,
    use_dedicated_endpoint: bool = False,
    max_num_seqs: int = 256,
    model_type: str = None,
    endpoint_path: str = None,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    
    
    if not endpoint_path:
        """Deploys trained models with vLLM into Vertex AI."""
        endpoint = aiplatform.Endpoint.create(
            display_name=f"{model_name}-endpoint",
            dedicated_endpoint_enabled=use_dedicated_endpoint,
        )

    else:
        endpoint = aiplatform.Endpoint(endpoint_path)


    if not base_model_id:
        base_model_id = model_id

    # See https://docs.vllm.ai/en/latest/models/engine_args.html for a list of possible arguments with descriptions.
    vllm_args = [
        "python",
        "-m",
        "vllm.entrypoints.api_server",
        "--host=0.0.0.0",
        "--port=8080",
        f"--model={base_model_bucket_path}",
        f"--tensor-parallel-size={accelerator_count}",
        "--swap-space=16",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-model-len={max_model_len}",
        f"--dtype={dtype}",
        f"--max-loras={max_loras}",
        f"--max-cpu-loras={max_cpu_loras}",
        f"--max-num-seqs={max_num_seqs}",
        "--disable-log-stats",
    ]

    if enable_trust_remote_code:
        vllm_args.append("--trust-remote-code")

    if enforce_eager:
        vllm_args.append("--enforce-eager")

    if enable_lora:
        vllm_args.append("--enable-lora")

    if enable_chunked_prefill:
        vllm_args.append("--enable-chunked-prefill")

    if enable_prefix_cache:
        vllm_args.append("--enable-prefix-caching")

    if 0 < host_prefix_kv_cache_utilization_target < 1:
        vllm_args.append(
            f"--host-prefix-kv-cache-utilization-target={host_prefix_kv_cache_utilization_target}"
        )

    if model_type:
        vllm_args.append(f"--model-type={model_type}")

    env_vars = {
        "MODEL_ID": model_id,
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": True,
        "DEPLOY_SOURCE": "notebook",
    }

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_args=vllm_args,
        serving_container_ports=[8080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=env_vars,
        serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB
        serving_container_deployment_timeout=7200,
    )

    print(model)

    print(
        f"Deploying {model_name} on {machine_type} with {accelerator_count} {accelerator_type} GPU(s)."
    )
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        deploy_request_timeout=1800,
        service_account=service_account
    )
    print("endpoint_name:", endpoint.name)

    return model, endpoint


if __name__ == "__main__":
    main()

