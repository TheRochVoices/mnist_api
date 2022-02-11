from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import cv2
from openvino.inference_engine import IECore
from torchvision import transforms

config = {'model_path': '/mnist_vino/mnist.xml'}
transform = transforms.ToTensor()
app = FastAPI()

def load_model():
	ie = IECore()
	net_ir = ie.read_network(model=config['model_path'])
	exec_net_ir = ie.load_network(network=net_ir, device_name="CPU")
	input_layer_ir = next(iter(exec_net_ir.input_info))
	output_layer_ir = next(iter(exec_net_ir.outputs))
	return [exec_net_ir, input_layer_ir, output_layer_ir]

net = load_model()

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
	contents = await image.read()
	pil_image = Image.open(io.BytesIO(contents))
	pil_image = pil_image.resize((28, 28))
	np_img = np.asarray(pil_image)
	np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)/255
	tensor = transform(np_img).unsqueeze(0)
	res_ir = net[0].infer(inputs={net[1]:tensor})
	res_ir = res_ir[net[2]]
	return {'predicted_num': str(res_ir.argmax())}

@app.get('/')
def root():
	print('aaaaaa')
	return {'message': 'inference api is up!'}

