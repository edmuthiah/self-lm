# DeepSeek R1 + LoRA on Vertex AI

Guide to set up your GCP to deploy `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` model with a LoRA adapter. Note you may need to raise QIR for sufficient GPUs.

---

### **Python Setup**
```python
python3 -m venv .venv
source .venv/bin/activate
pip3 install google-cloud-aiplatform huggingface-hub[cli]`
```

### **Cloud Setup**

ENV:
```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export SERVICE_ACCOUNT_NAME="vertex-deploy-sa"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export BUCKET_NAME="your-unique-bucket-name-here"
export BUCKET_URI="gs://${BUCKET_NAME}"
export HF_TOKEN="hf_your_hugging_face_read_token"
export BASE_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
export LORA_ADAPTER_GCS_PATH="${BUCKET_URI}/adaptors/custom-adaptor"
```

AUTH:
```
bash
gcloud auth login
glcoud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

CREATE:
```bash
# 1. Create a GCS Bucket
gsutil mb -l $REGION #BUCKET_URI

# 2. Create a Service Account
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME --display-name="Vertex AI Deployment Service Account"

# 3. Grant Permissions to the Service Account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Grant the SA permission to manage objects in your new bucket
gsutil iam ch \
    serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com:roles/storage.objectAdmin \
    gs://$BUCKET_NAME
```

### GCS Setup
Upload LoRA Adapter
```bash
hf auth login
hf download org/custom-adaptor --local-dir ./custom-adaptor

# Upload the local directory to your GCS bucket
gsutil -m cp -r ./custom-adaptor "${BUCKET_URI}/adaptors/"
```

Upload Base Model
```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-70B --local-dir ./deepseek-r1-70b-local --token $HF_TOKEN

# Upload the local directory to your GCS bucket
gsutil -m cp -r ./deepseek-r1-70b-local "${BUCKET_URI}/models/"
```

### Vertex AI Deployment
```python
python3 deploy.py
```

### Delete
```bash
gcloud ai endpoints delete YOUR_ENDPOINT_ID --region=$REGION --quiet
gcloud ai models delete YOUR_MODEL_ID --region=$REGION --quiet
```
