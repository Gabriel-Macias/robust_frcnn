import os
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--data_path", dest="data_path", help="Path to the KITTI dataset folder.")
parser.add_option("-d", "--domain_type", dest="domain_type", help="Domain type of the dataset: source (default) or target.", default="source")
parser.add_option("-c", "--classes_mode", dest="classes_mode", help="1: use custom object classes (default) 2: all classes", default=1, type=int)
parser.add_option("-t", "--change_classes", dest="change_classes", action = "store_true", help = "T: change class names to match source domain.", default=False)

(options, args) = parser.parse_args()

# If file data path is not given
if not options.data_path:
	parser.error('Error: path to dataset must be specified. Pass --data_path to command line')

# Selection of classes to be used
# All classes in KITTI: 
# ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

if options.classes_mode == 1:
	CLASSES_USED = ["Pedestrian"]
else:
	CLASSES_USED = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

if options.change_classes:
	# Example dictionary when using cityscapes as source and KITTI as target dataset only on pedestrians
	SOURCE_TARGET_DICT = {"Pedestrian": "person"}
	assert len(SOURCE_TARGET_DICT) > 0

IMAGES_DIR_NAME = "images/train"
ANNOT_DIR_NAME = "annotations"

# Output file must have "filename,x1,y1,x2,y2,class_name,data_type" format in each line

imgs_dir = os.path.join(options.data_path, IMAGES_DIR_NAME)
all_imgs = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]

if options.domain_type == "source":
	file_name = "source_kitti.txt"
	train_size = len(all_imgs)
else:
	file_name = "target_kitti.txt"
	train_size = int(0.8 * len(all_imgs))

np.random.seed(1232)
train_imgs = np.random.choice(all_imgs, train_size, replace = False).tolist()
test_imgs = np.setdiff1d(all_imgs, train_imgs).tolist()

img_sets = [train_imgs, test_imgs]
data_tags = ["train", "train"] if options.domain_type == "source" else ["train", "test"]

if options.domain_type == 2:
	train_tag = "test"
else:
	train_tag = "train"

# Creating the output file
with open(os.path.join(options.data_path, file_name), "w") as f:
	
	for data_tag, img_set in zip(data_tags, img_sets):
	
		for img in img_set:
			
			annot_file = img.replace(".png", ".txt").replace(IMAGES_DIR_NAME, ANNOT_DIR_NAME).replace("train/", "")
			
			with open(annot_file, "r") as f_txt:
				for line in f_txt:
					line_split = line.strip().split(" ")
					
					if line_split[0] in CLASSES_USED:
						x1, y1, x2, y2 = int(float(line_split[4])), int(float(line_split[5])), int(float(line_split[6])), int(float(line_split[7]))

						if options.change_classes:
							class_name = SOURCE_TARGET_DICT[line_split[0]]
						else:
							class_name = line_split[0]

						f.write("{},{},{},{},{},{},{}\n".format(img, x1, y1, x2, y2, class_name, data_tag))
					
			f_txt.close()
		
f.close()
