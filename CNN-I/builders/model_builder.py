import tensorflow as tf
import sys
sys.path.append("models")
from models.TinyFCN import build_tinyfcn

SUPPORTED_MODELS = ["TinyFCN"]



def build_model(model_name, net_input, num_classes, crop_width, crop_height, is_training=True):

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	network = None
	init_fn = None
	if model_name == "TinyFCN":
		network = build_tinyfcn(net_input, preset_model=model_name, num_classes=num_classes)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	print("Finish the model ...")
	return network, init_fn
