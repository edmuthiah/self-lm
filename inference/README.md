**1. Prerequisites:**
- `gcloud` SDK authenticated (`gcloud auth application-default login`)
- Python 3.8+ and `pip install google-cloud-aiplatform huggingface_hub[cli]`
- Git LFS (`git lfs install`)
- A GCS bucket

**2. Set Environment Variables:**
```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export BUCKET_URI="gs://your-bucket-name"
export SERVICE_ACCOUNT="your-service-account@your-project-id.iam.gserviceaccount.com"
export HF_TOKEN="hf_your_hugging_face_read_token"
export BASE_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
export LORA_ADAPTER_GCS_PATH="${BUCKET_URI}/loras/heidi-fast-v1p1-original"
```

**3. Prepare LoRA Adapter:**
```bash
# Clone the LoRA adapter from Hugging Face
hf auth login
hf download https://huggingface.co/org/custom-adaptor

# Upload it to your GCS bucket
gsutil -m cp -r ./custom-adaptor gs://your-bucket-name/adaptors/
```

**4. Prepare Base Model:**
```bash
# Download the base model from Hugging Face
huggingface-cli download \
  deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --local-dir ./deepseek-r1-70b-local \
  --token your_hf_token

# Upload it to your GCS bucket
gsutil -m cp -r ./deepseek-r1-70b-local gs://your-bucket-name/models/
```

**5. Run Script:**
```bash
python main.py
```
