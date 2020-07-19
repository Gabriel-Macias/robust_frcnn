import numpy as np
import cv2

def image_generator(data_file_path, C, mode = "source"):

	with open(data_file_path, "r") as f:
		all_img_data = list(map(lambda x: x.strip(), f.readlines()))
	f.close()
	
	n_classes = len(C.class_mapping)
	np.random.shuffle(all_img_data)

	new_height = 299
	new_width = 299

	while True:

		for img_data in all_img_data:
			try:
				# Output file must have "filename,x1,y1,x2,y2,class_name,data_type" format in each line
				if mode == "source":
					img_path, x1, y1, x2, y2, class_name, _ = img_data.split(",")
				else:
					img_path, x1, y1, x2, y2, class_name, probas = img_data.split(",")

				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				
				img = cv2.imread(img_path)
				cropped_img = img[y1:y2, x1:x2, :]
				
				x_resized = cv2.resize(np.copy(cropped_img), (int(new_width), int(new_height)), interpolation = cv2.INTER_CUBIC)
				x_resized = x_resized / 255.
				del img
				x = np.expand_dims(x_resized, axis = 0)
				
				if mode == "source":
					
					y = np.zeros((1, n_classes))
					
					if class_name in C.class_mapping.keys():
						y[0, C.class_mapping[class_name]] = 1
					else:
						y[0, -1] = 1
						
					yield x, y
					
				else:
					y = np.array(probas.split("|")).astype("float")
					yield x, np.expand_dims(y, axis = 0)
					
			except cv2.error as e:
				continue
			except Exception as e:
				print(e)
				continue
