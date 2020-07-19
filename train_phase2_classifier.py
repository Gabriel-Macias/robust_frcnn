from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model, model_from_json
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.utils import generic_utils
from keras_frcnn import phase3_utils
from optparse import OptionParser
import os
import pickle
from sklearn.metrics import f1_score

# Import image generator functions
from keras_frcnn import phase2_generator

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
		target_probas[idx_conf[0], :] = 0
		target_probas[idx_conf] = 1
		
		# Easy predictions (agreement between img and current)
		idx_agree = np.where((p_img[0].argmax(1) == p_curr[0].argmax(1)) & (p_curr[0].max(1) >= 0.50))[0]
		cols_agree = p_curr[0].argmax(1)[idx_agree]
		target_probas[idx_agree,:] = 0
		target_probas[idx_agree, cols_agree] = 1
		
		idx = np.unique(idx_conf[0].tolist() + idx_agree.tolist()).tolist()
	
	return np.expand_dims(target_probas, axis = 0), idx

parser = OptionParser()

parser.add_option("-s", "--source_path", dest="source_path", help="Path to the source txt file.")
parser.add_option("-t", "--target_path", dest="target_path", help="Path to the target detections txt file.")
parser.add_option("-o", "--original_detector_path", dest="original_detector_path", help="Path to the txt file used in phase 1.")
parser.add_option("-d", "--save_dir", dest="save_dir", help="Path to directory where architecture and weights will be saved.", default="models/phase2")
parser.add_option("-a", "--model_architecture", dest="model_architecture", help="Path to JSON where architecture will be saved (inside save_dir).", default="phase2_model.json")
parser.add_option("-w", "--model_weights", dest="model_weights", help="Path to .hdf5 where weights will be saved (inside save_dir).", default="phase2_weights.hdf5")
parser.add_option("-e", "--num_epochs", dest="num_epochs", help="Number of epochs for the training.", default=1, type=int)
parser.add_option("--e_length", dest="e_length", help="Epoch length - Steps for each epoch.", default=1000, type=int)
parser.add_option("--config_filename", dest="config_filename", help="Path of the config file of phase 1 F-RCNN.", default="config.pickle")
parser.add_option("-r", "--reg_param", dest="reg_param", help="Regularization parameter for semi-supervised training.", default=0.2, type=float)
parser.add_option("--sup_lr", dest="sup_lr", help="Learning rate used for the supervised training.", default=1e-5, type=float)
parser.add_option("--val_size", dest="val_size", help="Nb of images to use as val set to monitor performance and save weights (default 100).", default=100, type=int)

parser.add_option("-m", "--model_type", dest="model_type", help="Model to be used. 1: Noisy labels (default) 2: Entropy-minimization.", default=1, type=int)
parser.add_option("--alpha_init", type=float, dest="alpha_init", help="Starting alpha value for noisy-label model.", default=100.)
parser.add_option("--alpha_final", type=float, dest="alpha_final", help="Final/smallest alpha value for noisy-label model.", default=0.5)
parser.add_option("--hard_constraints", dest="hard_constraints", help="Set hard thresholds on confident predictions", action="store_true", default=False)
parser.add_option("--recompute_alpha", dest="recompute_alpha", help="Recompute alpha automatically using Hausdorf distance.", action="store_true", default=False)

(options, args) = parser.parse_args()

if not options.source_path:   # if filename is not given
	parser.error('Error: path to source dataset must be specified. Pass --source_path to command line')
if not options.target_path:
	parser.error('Error: path to target detections dataset must be specified. Pass --target_path to command line')

if not os.path.isdir(options.save_dir):
	os.mkdir(options.save_dir)

with open(options.config_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# Check the correct ordering
if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

# Define the number of classes - Background is included
n_classes = len(C.class_mapping)
print("Number of classes (used for phase 1) = {}".format(n_classes))
print("========== Creating architectures ==========")

base_cnn = InceptionV3(include_top = False, weights = 'imagenet', input_shape = input_shape_img)

x = base_cnn.output
x = GlobalAveragePooling2D(name = "final_globalavgpooling")(x)
x = Dense(4096, activation = 'relu', name = "final_dense1")(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation = 'relu', name = "final_dense2")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu', name = "final_dense3")(x)
x = Dropout(0.5)(x)
x = Dense(n_classes, activation = "softmax", name = "predictions")(x)

sup_img_classifier = Model(inputs = base_cnn.input, outputs = x)
if options.model_type == 2:
	semi_img_classifier = Model(inputs = base_cnn.input, outputs = x)

reg_param = options.reg_param
def semi_loss(y_true, y_pred):
	y_pred = y_pred + 1e-8
	return - reg_param * K.mean(y_pred * K.log(y_pred))

semi_sup_lr_ratio = 5
supervised_lr = options.sup_lr
semi_lr = supervised_lr / semi_sup_lr_ratio

optimizer_sup = Adam(lr = supervised_lr, clipnorm = 1e-2)
optimizer_semi = Adam(lr = semi_lr, clipnorm = 1e-4)

#optimizer_sup = RMSprop(supervised_lr)
#optimizer_semi = RMSprop(semi_lr)

#optimizer_sup = SGD(lr = supervised_lr, clipnorm = 1e-3, nesterov = False)
#optimizer_semi = SGD(lr = semi_lr, clipnorm = 1e-3, nesterov = False)

sup_img_classifier.compile(optimizer = optimizer_sup, loss = "categorical_crossentropy")
if options.model_type == 2:
	semi_img_classifier.compile(optimizer = optimizer_semi, loss = semi_loss)

# Saving the model architecture
with open(os.path.join(options.save_dir, options.model_architecture), "w") as f:
	f.write(sup_img_classifier.to_json())
f.close()

print("========== Created and saved architectures ========")

# We create the training generators
data_gen_source = phase2_generator.image_generator(options.source_path, C, mode = "source")
data_gen_original = phase2_generator.image_generator(options.original_detector_path, C, mode = "source")
data_gen_target = phase2_generator.image_generator(options.target_path, C, mode = "target")

sup_loss = np.zeros(options.e_length)
semi_loss = np.zeros(options.e_length)
n_epochs = options.num_epochs
start_time = time.time()
best_acc = -np.Inf
batch_size = 32

# Making the validation set to measure improvement
val_size = options.val_size
x_test, y_test = next(data_gen_source)
y_true = [y_test.argmax()]

for i in range(1, val_size):
	if i % 2 == 0:
		x_next, y_test = next(data_gen_original)
	else:
		x_next, y_test = next(data_gen_source)

	x_test = np.concatenate((x_test, x_next), axis = 0)
	y_true.append(y_test.argmax())

sup_loss_hist = []
semi_loss_hist = []
time_hist = []
f1_loss_hist = []

alpha_init = float(options.alpha_init)
alpha_final = float(options.alpha_final)
constant_thresh = int(5 / 7 * options.e_length * n_epochs)

print("========== Starting training ============")

# Begin the training
for epoch in range(n_epochs):
	progbar = generic_utils.Progbar(options.e_length)
	print('Epoch {}/{}'.format(epoch + 1, n_epochs))
	
	iter_num = 0

	if epoch > 0 and epoch % 3 == 0:
		supervised_lr = supervised_lr * 0.1
		#semi_lr = semi_lr * 0.94
		K.set_value(sup_img_classifier.optimizer.lr, supervised_lr)
		#K.set_value(semi_img_classifier.optimizer.lr, semi_lr)
	
	while True:
		try:
		
			if iter_num <= constant_thresh:
				alpha = alpha_init - iter_num * (alpha_init - alpha_final) / constant_thresh
			
			X_source, Y_source = next(data_gen_source)
			X_target, Y_target = next(data_gen_target)

			for b in range(1, batch_size):		
				if b % 3 == 0:
					x_next, y_next = next(data_gen_original)
				else:
					x_next, y_next = next(data_gen_source)

				X_source, Y_source = np.concatenate((X_source, x_next), axis = 0), np.concatenate((Y_source, y_next), axis = 0)
				
				x_next, y_next = next(data_gen_target)
				X_target = np.concatenate((X_target, x_next), axis = 0)
				
				if options.model_type == 1:
					Y_target = np.concatenate((Y_target, y_next), axis = 0)
			
			# Run one supervised step
			sup_loss[iter_num] = sup_img_classifier.train_on_batch(X_source, Y_source)
			
			# Run one semi-supervised step
			if options.model_type == 2:
				semi_loss[iter_num] = semi_img_classifier.train_on_batch(X_target, Y_source) # We pass Y_source because of Keras, but it's not used
			else:
				curr_probas = np.expand_dims(sup_img_classifier.predict(X_target), axis = 0)
				Y_target = np.expand_dims(Y_target, axis = 0)
				
				if options.recompute_alpha:
					alpha = get_optimal_alpha(Y_target, curr_probas, "max")
					
				Y_target, _ = make_target_probas(Y_target, curr_probas, alpha, constrain_hard_examples = options.hard_constraints)
				semi_loss[iter_num] = sup_img_classifier.train_on_batch(X_target, Y_target[0])
			
			progbar.update(iter_num, [('Supervised Loss', sup_loss[iter_num].mean()), ('Semi-Sup Loss', semi_loss[iter_num].mean()),
									  ("Total Loss", (sup_loss[iter_num] + semi_loss[iter_num]).mean())])

			iter_num += 1

			if iter_num == options.e_length:

				y_pred = sup_img_classifier.predict_on_batch(x_test).argmax(1).tolist()
				curr_acc = f1_score(y_true, y_pred, average = "micro")	

				semi_loss_hist.extend(semi_loss.tolist())
				sup_loss_hist.extend(sup_loss.tolist())
				time_hist.append(time.time() - start_time)
				f1_loss_hist.append(curr_acc)
				
				if C.verbose:
					print('\nSupervised Loss: {}'.format(sup_loss.mean()))
					print('Semi-Supervised Loss {}'.format(semi_loss.mean()))
					print('Total Loss: {}'.format(np.nanmean(sup_loss + semi_loss)))
					print("Current F1-Score: {}".format(curr_acc))
					print('Elapsed time: {}'.format(time.time() - start_time))

				if curr_acc > best_acc:
					if C.verbose:
						print('Total F1-Score increased from {} to {}, saving weights'.format(best_acc, curr_acc))
					best_acc = curr_acc
					sup_img_classifier.save_weights(os.path.join(options.save_dir, options.model_weights))
					
				start_time = time.time()

				break

		
		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print("=========== Finished training =============")
#np.savez("img_train_results.npz", sup_loss = sup_loss_hist, semi_loss = semi_loss_hist,  time_hist = time_hist, f1_loss = f1_loss_hist, n_epochs = n_epochs, epoch_length = options.e_length)
