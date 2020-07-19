import os
from optparse import OptionParser
import pickle
import time
import cv2
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model, model_from_json
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import generic_utils
from sklearn.metrics import f1_score

parser = OptionParser()

parser.add_option("-d", "--data_path", dest="data_path", help="Path to txt data file to perform the testing.")
parser.add_option("-a", "--model_architecture", dest="model_architecture", help="Path to JSON where architecture will be saved (inside save_dir).", default="phase2_model.json")
parser.add_option("-w", "--model_weights", dest="model_weights", help="Path to .hdf5 where weights will be saved (inside save_dir).", default="phase2_weights.hdf5")
parser.add_option("-n", "--num_samples", dest="num_samples", help="Number of samples used for testing (default 100).", type=int, default=100)
parser.add_option("--config_filename", dest="config_filename", help="Path of the config file of phase 1 F-RCNN.", default="config.pickle")

(options, args) = parser.parse_args()

# Load config file
with open(options.config_filename, "rb") as f:
	C = pickle.load(f)
f.close()

# load json and create model
json_file = open(options.model_json_path, 'r')
img_classifier = model_from_json(json_file.read())
json_file.close()

# load weights into new model
img_classifier.load_weights(options.weight_path)
print("Loaded model from disk")

# Load X testing cropped images 
with open(options.data_path, "r") as f:
	all_lines = list(map(lambda x: x.strip(), f.readlines()))
f.close()

if options.num_samples > 0:
	np.random.shuffle(all_lines)
	all_lines = all_lines[:int(options.num_samples)]

class_map = C.class_mapping

new_height = 299
new_width = 299
y_true = []
y_pred = []

for line in all_lines:
	
	try:
		img_path, x1, y1, x2, y2, class_name, _ = line.split(",")
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		
		img = cv2.imread(img_path)
		cropped_img = img[y1:y2, x1:x2, :]

		x_resized = cv2.resize(np.copy(cropped_img), (int(new_width), int(new_height)), interpolation = cv2.INTER_CUBIC)
		x_resized = x_resized / 255.
		x = np.expand_dims(x_resized, axis = 0)
		
		pred = img_classifier.predict_on_batch(x)
		y_pred.append(pred.argmax())

		if class_name in class_map.keys():
			y_true.append(class_map[class_name])
		else:
			y_true.append(len(class_map) - 1)
		
		if pred[0][0] < 0.5 and class_name == "person":
			print("Prediction on img {} for label {} = {} <======= MISTAKE".format(img_path, class_name, pred[0][0]))
		elif pred[0][0] > 0.5 and class_name != "person":
			print("Prediction on img {} for label {} = {} <======= MISTAKE".format(img_path, class_name, pred[0][0]))
		#else:
		#	print("Prediction on img {} for label {} = {}".format(img_path, class_name, pred[0][0]))
	
	except Exception as e:
		print(e)
		continue

print("============ FINISHED ===========")
print("Final Accuracy = {}".format(np.mean(np.array(y_true) == np.array(y_pred))))
print("Final F1-Score = {}".format(f1_score(y_true, y_pred, average = "micro")))

