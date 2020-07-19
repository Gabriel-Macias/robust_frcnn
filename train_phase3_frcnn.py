from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import cv2
from optparse import OptionParser
import pickle
import os
import traceback

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model, load_model, model_from_json
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

if 'tensorflow' == K.backend():
	import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

sys.setrecursionlimit(40000)

def kl_div(P, Q):
	return np.nansum([p * np.log2(p / (q + 1e-8)) for p, q in zip(P, Q) if p != 0])

def js_distance(P, Q):
	M = 0.5 * (P + Q)
	return np.sqrt(0.5 * kl_div(P, M) + 0.5 * kl_div(Q, M))

def get_optimal_alpha(p_img, p_curr, rule_mode = "max"):

	js_dist_list = [js_distance(p_img[0,i,:], p_curr[0,i,:]) for i in range(p_img.shape[1])]

	if rule_mode == "max":
		dist_diff = np.nanmax(js_dist_list)
	elif rule_mode == "min":
		dist_diff = np.nanmin(js_dist_list)
	else:
		dist_diff = np.nanmean(js_dist_list)

	return np.max([alpha_final, dist_diff / (1 - dist_diff + 1e-8)])

def make_target_probas(p_img, p_curr, alpha, constrain_hard_examples = False):
	target_probas = (np.log(p_curr[0] + 1e-8) + alpha * np.log(p_img[0] + 1e-8)) / (1 + alpha)
	target_probas = np.exp(target_probas) / np.exp(target_probas).sum(axis = 1)[:, None]
	idx = []

	if constrain_hard_examples:

		# Confident predictions in img_classifier
		idx_conf = np.where(p_img[0] >= 0.90)
		target_probas[idx_conf[0],:] = 0
		target_probas[idx_conf] = 1

		# Easy predictions (agreement between img and current)
		idx_agree = np.where((p_img[0].argmax(1) == p_curr[0].argmax(1)) & (p_curr[0].max(1) >= 0.50))[0]
		cols_agree = p_curr[0].argmax(1)[idx_agree]
		target_probas[idx_agree,:] = 0
		target_probas[idx_agree, cols_agree] = 1

		idx = np.unique(idx_conf[0].tolist() + idx_agree.tolist()).tolist()

	return np.expand_dims(target_probas, axis = 0), idx

def make_target_bbs(bb_curr, bb_phase1, alpha):
	target_bbs = (bb_curr + alpha * bb_phase1) / (1 + alpha)
	return target_bbs

def get_img_probas(img_path, P_cls, P_regr, ROIs, C, f):

	img = cv2.imread(img_path)
	new_height = 299
	new_width = 299
	img_probas = np.zeros((P_cls.shape[1], len(class_mapping)))

	for ii in range(P_cls.shape[1]):

		(x, y, w, h) = ROIs[0, ii, :]
		cls_num = np.argmax(P_cls[0, ii, :])

		try:
			(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
			tx /= C.classifier_regr_std[0]
			ty /= C.classifier_regr_std[1]
			tw /= C.classifier_regr_std[2]
			th /= C.classifier_regr_std[3]
			x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
		except:
			pass

		# Get the true BB coordinates
		x1, y1, x2, y2 = C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)
		x1, y1, x2, y2 = data_generators.get_real_coordinates(f, x1, y1, x2, y2)

		# Get the probabilities from the image classifier
		cropped_img = img[y1:y2, x1:x2, :]
		x_resized = cv2.resize(np.copy(cropped_img), (int(new_width), int(new_height)), interpolation = cv2.INTER_CUBIC)
		x_resized = x_resized / 255.
		x_resized = np.expand_dims(x_resized, axis = 0)

		img_probas[ii, :] = img_classifier.predict(x_resized)[0]

	return np.expand_dims(img_probas, axis = 0)

def rpn_to_class_inputs(X, img_data, C, mode = "source", eps = 0.05):

	[Y1, Y2] = model_rpn.predict(X)
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), use_regr = True, overlap_thresh = 0.4, max_boxes = 300)

	X2, Y1, Y2, _ = roi_helpers.calc_iou(R, img_data, C, class_mapping, mode, eps)

	if X2 is None:
		rpn_accuracy_rpn_monitor.append(0)
		rpn_accuracy_for_epoch.append(0)
		raise NameError('No quality ROIs in X2. Training on another sample')

	neg_samples = np.where(Y1[0, :, :].argmax(1) == len(class_mapping) - 1)
	pos_samples = np.where(Y1[0, :, :].argmax(1) != len(class_mapping) - 1)

	if len(neg_samples) > 0:
		neg_samples = neg_samples[0]
	else:
		neg_samples = []

	if len(pos_samples) > 0:
		pos_samples = pos_samples[0]
	else:
		pos_samples = []

	rpn_accuracy_rpn_monitor.append(len(pos_samples))
	rpn_accuracy_for_epoch.append((len(pos_samples)))

	if C.num_rois > 1:
		if len(pos_samples) < C.num_rois//2:
			selected_pos_samples = pos_samples.tolist()
		else:
			selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
		try:
			selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
		except:
			selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
		sel_samples = selected_pos_samples + selected_neg_samples
	else:
		# In the extreme case where num_rois = 1, we pick a random pos or neg sample
		selected_pos_samples = pos_samples.tolist()
		selected_neg_samples = neg_samples.tolist()
		if np.random.randint(0, 2):
			sel_samples = random.choice(neg_samples)
		else:
			sel_samples = random.choice(pos_samples)

	X2 = X2[:, sel_samples, :]
	Y1 = Y1[:, sel_samples, :]
	Y2 = Y2[:, sel_samples, :]

	return X2, Y1, Y2, len(selected_pos_samples)

def get_target_img_data(X_target, img_data, alpha, constrain_hard_examples = False, use_optimal_alpha = False):

	[Y1, Y2, F] = phase1_rpn.predict(X_target)
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh = 0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}
	all_probs = {}

	for jk in range(R.shape[0] // C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis = 0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0] // C.num_rois:
			# Pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		# Make predictions with current FRCNN and phase 1 detector
		[_, P_regr_phase1] = phase1_classifier.predict([F, ROIs])
		[P_cls_curr, P_regr_curr] = model_classifier.predict([X_target, ROIs]) # <- This returns a (1, n_ROIs, n_class) and (1, n_ROIs, 4) tensors

		# Get the probabilities from the image classifier
		img_probas = get_img_probas(filepath, P_cls_curr, P_regr_curr, ROIs, C, f)

		# Optional re-computation of the alpha parameter
		if use_optimal_alpha:
			alpha = get_optimal_alpha(img_probas, P_cls_curr, "mean")

		# Get the target probabilities
		P_cls, no_change_bb_idx = make_target_probas(img_probas, P_cls_curr, alpha, constrain_hard_examples)

		for ii in range(P_cls.shape[1]):

			# If the detected object is bg skip
			if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = inv_map[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []
				all_probs[cls_name] = []

			cls_num = np.argmax(P_cls[0, ii, :])
			(x1, y1, w1, h1) = ROIs[0, ii, :]
			(x2, y2, w2, h2) = ROIs[0, ii, :]

			try:
				(tx, ty, tw, th) = P_regr_phase1[0, ii, 4 * cls_num:4 * (cls_num + 1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x1, y1, w1, h1 = roi_helpers.apply_regr(x1, y1, w1, h1, tx, ty, tw, th)
			except:
				pass

			try:
				(tx, ty, tw, th) = P_regr_curr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x2, y2, w2, h2 = roi_helpers.apply_regr(x2, y2, w2, h2, tx, ty, tw, th)
			except:
				pass

			if ii in no_change_bb_idx:
				x, y, w, h = x2, y2, w2, h2
			else:
				x, y, w, h = make_target_bbs(np.array([x2, y2, w2, h2]), np.array([x1, y1, w1, h1]), alpha)

			bboxes[cls_name].append([C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))
			all_probs[cls_name].append(P_cls[0, ii, :])

	for key in bboxes:

		new_boxes, _, chosen_idx = roi_helpers.non_max_suppression_fast(np.array(bboxes[key]), np.array(probs[key]), overlap_thresh = 0.1)
		probas = np.array(all_probs[key])[chosen_idx, :]

		# img_data = {"filepath" : filepath, "width" : width, "height" : height, "bboxes" : []}
		# all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

		for jk in range(new_boxes.shape[0]):

			(x1, y1, x2, y2) = new_boxes[jk, :]
			(x1, y1, x2, y2) = data_generators.get_real_coordinates(f, x1, y1, x2, y2)

			img_data["bboxes"].append({'class': key, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'probas': probas[jk, :]})

	return img_data

parser = OptionParser()

parser.add_option("-s", "--source_path", dest="source_path", help="Path to source training txt file.")
parser.add_option("-t", "--target_path", dest="target_path", help="Path to target training detections txt file.")
parser.add_option("-p", "--parser", dest="parser", help="Parser to use. One of general or pascal_voc", default="general")
parser.add_option("-r", "--num_rois", type="int", dest="num_rois", help="Number of ROIs to process at once.", default=32)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=50)
parser.add_option("--elen", dest="epoch_length", help="Set the epoch length. def=1000", default=1000)
parser.add_option("--opt", dest="optimizers", help="Set the optimizer to use", default="SGD")
parser.add_option("--lr", dest="lr", help="Initial learning rate", type=float, default=1e-3)
parser.add_option("--load_checkpoint", dest="load_checkpoint", help="Path to model weights from past checkpoint. Used to resume training.", default=None)

parser.add_option("--alpha_init", type=float, dest="alpha_init", help="Starting alpha value.", default=100.)
parser.add_option("--alpha_final", type=float, dest="alpha_final", help="Final/smallest alpha value.", default=0.5)
parser.add_option("--hard_constraints", dest="hard_constraints", help="Set hard thresholds on confident predictions", action="store_true", default=False)
parser.add_option("--recompute_alpha", dest="recompute_alpha", help="Recompute alpha automatically using Hausdorf distance.", action="store_true", default=False)

parser.add_option("--phase1_config_file", dest="phase1_config", help="Path of the config file of phase 1 F-RCNN.", default="config.pickle")
parser.add_option("--phase1_weights", dest="phase1_weights", help="Path to .hdf5 file with phase 1 F-RCNN model weights")
parser.add_option("--img_json", dest="img_json_path", help="Path to JSON file with phase 2 img model architecture")
parser.add_option("--img_weights", dest="img_weight_path", help="Path to .hdf5 file with phase 2 img model weights")

parser.add_option("--output_config_file", dest="output_config", help="Path to save final phase 3 config file (for testing)", default="config_phase3.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='models/phase3/phase3_weights.hdf5')


(options, args) = parser.parse_args()

# Check for user errors
if not options.phase1_weights:
	parser.error('Error: path to phase 1 weights must be specified. Pass --phase1_weights to command line')
if not options.img_json_path:
	parser.error('Error: path to phase 2 JSON file must be specified. Pass --img_json to command line')
if not options.img_weight_path:
	parser.error('Error: path to phase 2 weights must be specified. Pass --img_weights to command line')
if not options.source_path:
	parser.error('Error: path to source training data must be specified. Pass --source_path to command line')
if not options.target_path:
	parser.error('Error: path to target training data must be specified. Pass --target_path to command line')

# Loading the selected parser
if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == "general":
	from keras_frcnn.general_parser import get_data
else:
	raise ValueError("Command line option parser must be a valid one")

# mkdir to save models.
if not os.path.isdir("models"):
	os.mkdir("models")
if not os.path.isdir("models/phase3"):
	os.mkdir("models/phase3")

# Loading the config file from phase 1
with open(options.phase1_config, 'rb') as f_in:
	C = pickle.load(f_in)

C.num_rois = int(options.num_rois)
C.model_path = options.output_weight_path

# Select the proper backbone configuration
if C.network == 'vgg16':
	from keras_frcnn import vgg as nn
	feature_dim = 512
elif C.network == 'resnet50':
	from keras_frcnn import resnet as nn
	feature_dim = 1024
elif C.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	feature_dim = 512
elif C.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	feature_dim = 512
elif C.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	feature_dim = 320
elif C.network == 'densenet':
	from keras_frcnn import densenet as nn
	feature_dim = 1024
else:
	print('Check network name in phase 1 config file.')
	raise ValueError

# Load source and target data and creating the generators
source_imgs, classes_count, _ = get_data(options.source_path)
target_imgs, _, _ = get_data(options.target_path)

class_mapping = C.class_mapping

if 'bg' not in classes_count:
	classes_count['bg'] = 0

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

inv_map = {v: k for k, v in class_mapping.items()}

print('Source training images per class:')
pprint.pprint(classes_count)
print('Num source classes (including bg) = {}'.format(len(classes_count)))

with open(options.output_config, 'wb') as config_f:
	pickle.dump(C, config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(options.output_config))

source_train_imgs = [s for s in source_imgs if s['imageset'] == 'train']
target_train_imgs = [s for s in target_imgs if s['imageset'] == 'train']
source_val_imgs = [s for s in source_imgs if s['imageset'] == 'test'] # Feeling pretty, might delete later

random.shuffle(source_train_imgs)
random.shuffle(source_val_imgs)
random.shuffle(target_train_imgs)

print('Num source train images {}'.format(len(source_train_imgs)))
#print('Num source val images {}'.format(len(source_val_imgs)))
print('Num target train images {}'.format(len(target_train_imgs)))

data_gen_source_train = data_generators.get_anchor_gt(source_train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode = 'train')
#data_gen_source_val = data_generators.get_anchor_gt(source_val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode = 'val')
data_gen_target_train = data_generators.get_anchor_gt(target_train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode = 'val')

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (feature_dim, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, feature_dim)

# Loading the phase 1 detector
img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (C.num_rois, 4))
feature_map_input = Input(shape = input_shape_features)

shared_layers = nn.nn_base(img_input, trainable = True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes = len(class_mapping), trainable = True)

phase1_rpn = Model(img_input, rpn_layers)
phase1_classifier = Model([feature_map_input, roi_input], classifier)

phase1_rpn.load_weights(options.phase1_weights, by_name = True)
phase1_classifier.load_weights(options.phase1_weights, by_name = True)

phase1_rpn.compile(optimizer = 'sgd', loss = 'mse')
phase1_classifier.compile(optimizer = 'sgd', loss = 'mse')
print("Loaded phase 1 Faster R-CNN detector")

# Loading the image classifier
# load json and create model
json_file = open(options.img_json_path, 'r')
img_classifier = model_from_json(json_file.read())
json_file.close()

# load weights into new model
img_classifier.load_weights(options.img_weight_path)
print("Loaded phase 2 image classifier")

# Creating the phase 3 detector
img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (None, 4))

shared_layers = nn.nn_base(img_input, trainable = True)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes = len(classes_count), trainable = True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# This is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# Load pretrained Imagenet weights
try:
	print('Loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name = True)
	model_classifier.load_weights(C.base_net_weights, by_name = True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# Use this to resume from previous training. Specify the frcnn model to load
if options.load_checkpoint is not None:
	print("Loading previous model from", options.load_checkpoint)
	model_rpn.load_weights(options.load_checkpoint, by_name = True)
	model_classifier.load_weights(options.load_checkpoint, by_name = True)
else:
	print("No previous model checkpoint was loaded")

# Optimizer setup
clipnorm_val = 1e-5
lr_val = options.lr
if options.optimizers == "SGD":
	optimizer = SGD(lr = lr_val, momentum = 0.9, clipnorm = clipnorm_val)
	optimizer_classifier = SGD(lr = lr_val, momentum = 0.9, clipnorm = clipnorm_val)
else:
	optimizer = Adam(lr = lr_val, clipnorm = clipnorm_val)
	optimizer_classifier = Adam(lr = lr_val, clipnorm = clipnorm_val / 1)

# Compile the model AFTER loading weights!
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer = 'sgd', loss = 'mae')

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}

# Hyperparameters of the robust F-RCNN
eps = 0.05
alpha_init = float(options.alpha_init)
alpha_final = float(options.alpha_final)
constant_thresh = int(5 / 7 * epoch_length * num_epochs)
iter_count = 0

print('Starting training')

for epoch_num in range(num_epochs):

	start_time = time.time()
	progbar = generic_utils.Progbar(epoch_length, stateful_metrics = ["rpn_cls", "rpn_regr", "detector_cls", "detector_regr", "avg nb of objects"])
	print('Epoch {} / {}'.format(epoch_num + 1, num_epochs))

#	if epoch_num > 0 and epoch_num < 45:
#		clipnorm_val = np.array(clipnorm_val * 0.95)
#		lr_val = lr_val * 0.95
#		K.set_value(model_rpn.optimizer.lr, lr_val)
#		K.set_value(model_classifier.optimizer.lr, lr_val)
#		K.set_value(model_rpn.optimizer.clipnorm, clipnorm_val)
#		K.set_value(model_classifier.optimizer.clipnorm, clipnorm_val)

	while True:
		try:

			if iter_count <= constant_thresh:
				alpha = alpha_init - iter_count * (alpha_init - alpha_final) / constant_thresh

			if iter_count == constant_thresh and options.load_checkpoint is None:
				lr_val = lr_val * 0.1
				K.set_value(model_rpn.optimizer.lr, lr_val)
				K.set_value(model_classifier.optimizer.lr, lr_val)

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('\nAverage number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))

				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			# Get next batch samples
			X, Y, img_data = next(data_gen_source_train)
			#X, Y, img_data = next(data_gen_source_val)

			# Unaltered RPN training with source data
			loss_rpn = model_rpn.train_on_batch(X, Y)

			# NOTE: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			# Y1 is the output with the one-hot hard labels [0,0,0,0,1,0]
			# X2 is the 1 x R x 4 tensor with the ROI coordinates to be trained, they're already in (x1,y1,w,h) format
			X2, Y1, Y2, n_pos_samples_1 = rpn_to_class_inputs(X, img_data, C, mode = "source")

			loss_class_1 = model_classifier.train_on_batch([X, X2], [Y1, Y2])

			# VERY IMPORTANT: This loop guarantees that there will always be one target step per source step
			while True:
				try:
					X_target, filepath, width, height, f = next(data_gen_target_train)

					img_data = {"filepath" : filepath, "width" : width, "height" : height, "bboxes" : []}
					img_data = get_target_img_data(X_target, img_data, alpha, options.hard_constraints, options.recompute_alpha)

					X2, Y1, Y2, n_pos_samples_2 = rpn_to_class_inputs(X_target, img_data, C, mode = "target", eps = eps)

					loss_class_2 = model_classifier.train_on_batch([X_target, X2], [Y1, Y2])

					break

				except Exception as e:
					#print(traceback.format_exc())
					#print('Exception: {} at line {}'.format(e, sys.exc_info()[2].tb_lineno))
					continue

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class_1[1] + loss_class_2[1]
			losses[iter_num, 3] = loss_class_1[2] + loss_class_2[2]
			losses[iter_num, 4] = np.mean([loss_class_1[3], loss_class_2[3]])

			progbar.update(iter_num, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
									  ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3]),
									  ("avg nb of objects", np.mean([n_pos_samples_1, n_pos_samples_2]))])

			iter_num += 1
			iter_count += 1

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4]).round(1)

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('\nMean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Total Loss: {}'.format(curr_loss))
					print('Elapsed time: {}'.format(time.time() - start_time))

				iter_num = 0

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

		except Exception as e:
			#print(traceback.format_exc())
			#print('Exception: {} at line {}'.format(e, sys.exc_info()[2].tb_lineno))
			continue

print('Training complete, exiting.')


