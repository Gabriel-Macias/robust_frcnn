import json
import os
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--data_path", dest="data_path", help="Path to the cityscapes dataset folder.")
parser.add_option("-d", "--domain_type", dest="domain_type", help="Domain type of the dataset: source (default) or target.", default="source")
parser.add_option("-c", "--classes_mode", dest="classes_mode", help="1: use custom object classes (default) 2: all classes", default=1, type=int)
parser.add_option("-t", "--change_classes", dest="change_classes", action = "store_true", help = "T: change class names to match source domain.", default=False)

(options, args) = parser.parse_args()

# If file data path is not given
if not options.data_path:
	parser.error('Error: path to dataset must be specified. Pass --data_path to command line')

# Selection of classes to be used
# All classes in cityscapes: 
# ['bicycle', 'bicyclegroup', 'bridge', 'building', 'bus', 'car', 'caravan', 'cargroup', 'dynamic', 'ego vehicle', 
#  'fence', 'ground', 'guard rail', 'license plate', 'motorcycle', 'motorcyclegroup', 'out of roi', 'parking', 'person', 
#  'persongroup', 'pole', 'polegroup', 'rail track', 'rectification border', 'rider', 'ridergroup', 'road', 'sidewalk', 
#  'sky', 'static', 'terrain', 'traffic light', 'traffic sign', 'trailer', 'train', 'truck', 'truckgroup', 'tunnel', 'vegetation', 'wall']

if options.classes_mode == 1:
	CLASSES_USED = ["person"]
else:
	CLASSES_USED = ['bicycle', 'bicyclegroup', 'bridge', 'building', 'bus', 'car', 'caravan', 'cargroup', 'dynamic', 'ego vehicle', 
					'fence', 'ground', 'guard rail', 'license plate', 'motorcycle', 'motorcyclegroup', 'out of roi', 'parking', 'person', 
					'persongroup', 'pole', 'polegroup', 'rail track', 'rectification border', 'rider', 'ridergroup', 'road', 'sidewalk', 
					'sky', 'static', 'terrain', 'traffic light', 'traffic sign', 'trailer', 'train', 'truck', 'truckgroup', 'tunnel', 
					'vegetation', 'wall']

if options.change_classes:
	# Example dictionary when using KITTI as source and cityscapes as target dataset only on pedestrians
	SOURCE_TARGET_DICT = {"person": "Pedestrian"}
	assert len(SOURCE_TARGET_DICT) > 0

IMAGES_DIR_NAME = "images"
ANNOT_DIR_NAME = "annotations"
city_dir_list = ["train", "val"]

if options.domain_type == "source":
	file_name = "source_cityscapes.txt"
else:
	file_name = "target_cityscapes.txt"

# Output file must have "filename,x1,y1,x2,y2,class_name,data_type" format in each line
# Opening the output file
with open(os.path.join(options.data_path, file_name), "w") as f:

	# For loop for train and val datasets
	for d_class in city_dir_list:
		
		imgs_dir = os.path.join(options.data_path, IMAGES_DIR_NAME, d_class)
		all_imgs = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]
		set_class = "train" if options.domain_type == "source" else d_class
		
		for img in all_imgs:
			
			annot_file = img.replace("leftImg8bit.png", "gtFine_polygons.json").replace(IMAGES_DIR_NAME, ANNOT_DIR_NAME)
			
			with open(annot_file, "r") as f_json:
				json_dict = json.load(f_json)
			f_json.close()
			
			for obj in json_dict["objects"]:
				if obj["label"] in CLASSES_USED:
					x_min, y_min = np.min(obj["polygon"], 0)
					x_max, y_max = np.max(obj["polygon"], 0)

					if options.change_classes:
						class_name = SOURCE_TARGET_DICT[obj["label"]]
					else:
						class_name = obj["label"]
					
					# If the object in the image corresponds to a class of interest write it to the output file
					f.write("{},{},{},{},{},{},{}\n".format(img, x_min, y_min, x_max, y_max, class_name, set_class))

f.close()
