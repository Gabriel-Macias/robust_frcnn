import sys
import time
import numpy as np
import cv2
import pickle
import os
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers

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
			alpha = get_optimal_alpha(img_probas, P_cls_curr, "max")
		
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