# Readme
##### Training:
To train a model, run the train.py with the relevant flags.
##### Converting model to openvino:
Since I am hosting the api on an intel cpu, i decided to convert it into intel's openvino format. If you are running on specialized hardware (gpus, nvidia boards), I'll recommend using tensorRT.
You can convert pytorch model to openvino using the pytroch2openvino.py

##### openvino inference
See the codes/inference_api.py for the REST api + openvino inference code
##### Building the Docker Image
To build the docker image, just run 'docker build -t mnist:test .', it builds the docker image named 'mnist' and tags it with test.
##### Hosting a api
Once the image is build, run 'docker run -p 80:80 mnist:test', it runs the api on port 80 inside the container, and port forwards that to the host machine's port 80.
##### Testing
Do
curl -k -X POST -F "image=@im.jpg" -v http://127.0.0.1:80/predict
on your host machine. It uploads the image in the query parameter "image", and recieves the prediction in the response. Refer to the api_response.jpg in the repo.
##### Kubernetes deployment
I have also added a sample kuberenetes deployment file which fetches the mnist:test image from dockerhub and and hosts it on the same port.

To get the least latency, highest throughput, I will recommend doing fp16 inference, using a framework like torchserve, hosted on an autoscalable k8s cluster. Torchserve batches  the incoming request which utilies the batch inference of the model, which is not possible if you have just hosted a model, which processes requests 1 by 1. 
