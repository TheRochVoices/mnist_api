import torch
import torchvision.transforms as transforms
import cv2
from main import Net
from openvino.inference_engine import IECore
import os
from torchvision import transforms

def load_model_to_mem(path):
	net = Net()
	net.load_state_dict(torch.load(path))
	net.eval()
	return net
def process_image(path):
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (28,28))
	transform = transforms.ToTensor()
	tensor = transform(image)
	return tensor.unsqueeze(0)
if __name__ == '__main__':
	ONNX_PATH = 'mnist.onnx'
	VINO_PATH = 'mnist_vino'

	net = load_model_to_mem('mnist_cnn.pt')
	x = torch.ones((1, 1, 28, 28))
	torch.onnx.export(net, x, ONNX_PATH, opset_version=11, do_constant_folding=False)
	mo_command = f"""mo --input_model "{ONNX_PATH}" --input_shape "[1,1, 28, 28]" --data_type FP16 --output_dir {VINO_PATH}"""
	os.system(mo_command)	
