import os
import numpy as np
import cv2
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--data_path", dest="data_path", help="Path to the GTA5 dataset folder.")
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
	CLASSES_USED = ["person", "car"]
else:
	CLASSES_USED = ['bicycle', 'bridge', 'building', 'bus', 'car', 'caravan', 'dynamic', 'ego vehicle',
					'fence', 'ground', 'guard rail', 'license plate', 'motorcycle', 'out of roi', 'parking', 'person', 
					'pole', 'polegroup', 'rail track', 'rectification border', 'rider', 'road', 'sidewalk', 
					'sky', 'static', 'terrain', 'traffic light', 'traffic sign', 'trailer', 'train', 'truck', 'tunnel', 
					'vegetation', 'wall']

if options.change_classes:
	# Example dictionary when using KITTI as source and GTA as target dataset on only pedestrians
	SOURCE_TARGET_DICT = {"person": "Pedestrian"}
	assert len(SOURCE_TARGET_DICT) > 0

IMAGES_DIR_NAME = "images"
ANNOT_DIR_NAME = "labels"

# Output file must have "filename,x1,y1,x2,y2,class_name,data_type" format in each line

imgs_dir = os.path.join(options.data_path, IMAGES_DIR_NAME)
all_imgs = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]

if options.domain_type == "source":
	file_name = "source_gta.txt"
	train_size = len(all_imgs)
else:
	file_name = "target_gta.txt"
	train_size = int(0.8 * len(all_imgs))

np.random.seed(1232)
train_imgs = np.random.choice(all_imgs, train_size, replace = False).tolist()
test_imgs = np.setdiff1d(all_imgs, train_imgs).tolist()

img_sets = [train_imgs, test_imgs]
data_tags = ["train", "train"] if options.domain_type == "source" else ["train", "test"]

class_dict = {}
with open(os.path.join(options.data_path, "map_file.txt"), "r") as f:
	for line in f:
		split = line.strip().split(",")
		class_dict[split[0]] = np.array(split[1:]).astype("int")
f.close()

# Creating the output file
with open(os.path.join(options.data_path, file_name), "w") as f:
	
	for data_tag, img_set in zip(data_tags, img_sets):
	
		for img_path in img_set:
			
			label_file = img_path.replace(IMAGES_DIR_NAME, ANNOT_DIR_NAME)
			
			img = cv2.imread(label_file)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			for obj_class in CLASSES_USED:

				if options.change_classes:
					class_name = SOURCE_TARGET_DICT[obj_class]
				else:
					class_name = obj_class
				
				thresh = cv2.inRange(img, class_dict[obj_class], class_dict[obj_class])

				# Find contours, obtain bounding box, extract and save ROI
				cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				cnts = cnts[0] if len(cnts) == 2 else cnts[1]

				for c in cnts:
					x1, y1, w, h = cv2.boundingRect(c)
					if w >= 15 and h >= 15:
						x2, y2 = x1 + w, y1 + h
						x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
						f.write("{},{},{},{},{},{},{}\n".format(img_path, x1, y1, x2, y2, class_name, data_tag))
		
f.close()
