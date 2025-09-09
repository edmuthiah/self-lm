# Source: https://raw.githubusercontent.com/googleapis/python-aiplatform/refs/heads/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create Endpoint backing LRO: projects/738744999395/locations/us-central1/endpoints/2250272592034267136/operations/1867519554884730880
Endpoint created. Resource name: projects/738744999395/locations/us-central1/endpoints/2250272592034267136
To use this Endpoint in another session:
endpoint = aiplatform.Endpoint('projects/738744999395/locations/us-central1/endpoints/2250272592034267136')
Creating Model
Create Model backing LRO: projects/738744999395/locations/us-central1/models/5340753487107981312/operations/9126759204252549120
"""

# [START aiplatform_predict_custom_trained_model_sample]
from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    lora_id: str = None,
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]

    parameters = {}
    if lora_id:
        parameters = {'lora_id': lora_id}

    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", prediction)


# [END aiplatform_predict_custom_trained_model_sample]


instances = [{
    "prompt": "question does here",
    "max_tokens": 1024,
    "temperature": 0.2
}]

predict_custom_trained_model_sample(
    project=PROJECT_ID,
    endpoint_id=ENDPOINT_ID,
    location="us-central1",
    instances=instances,
    lora_id=LORA_GCS_URI
)
