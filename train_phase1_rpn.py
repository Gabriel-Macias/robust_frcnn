# -*- coding: utf-8 -*-
# stuff
from __future__ import division
import random
import pprint
import keras
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import data_generators
from keras_frcnn import config
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

# gpu setting
if 'tensorflow' == K.backend():
    import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

# command arg example
#--network mobilenetv1 -o simple -p ../../datasets/small_kitti_obj/kitti_train.txt

# option parsar
parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of general or pascal_voc", default="general")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=10)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='vgg19')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=50)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).", default="config.pickle")
parser.add_option("--elen", dest="epoch_length", help="set the epoch length. def=1000", type="int", default=1000)
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

# make dirs to save rpn
# "./models/rpn/rpn"
if not os.path.isdir("models"):
	os.mkdir("models")
if not os.path.isdir("models/rpn"):
	os.mkdir("models/rpn")

# we will train from pascal voc 2007
# you have to pass the directory of VOC with -p
if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'general':
	from keras_frcnn.general_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

# set data argumentation
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

# we will use resnet. may change to vgg
if options.network == 'vgg':
	C.network = 'vgg16'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	C.network = 'vgg19'
elif options.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	C.network = 'mobilenetv1'
elif options.network == 'mobilenetv1_05':
	from keras_frcnn import mobilenetv1_05 as nn
	C.network = 'mobilenetv1_05'
elif options.network == 'mobilenetv1_25':
	from keras_frcnn import mobilenetv1_25 as nn
	C.network = 'mobilenetv1_25'
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
elif options.network == 'densenet':
	from keras_frcnn import densenet as nn
	C.network = 'densenet'
else:
	print('Not a valid model')
	raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
	C.base_net_weights = options.input_weight_path
else:
	# set the path to weights based on backend and model
	C.base_net_weights = nn.get_weight_path()


# place weight files on your directory
base_net_weights = nn.get_weight_path()


#### load images here ####
# get voc images
all_imgs, classes_count, class_mapping = get_data(options.train_path)

print(classes_count)

# add background class as 21st class
if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

# split to train and val
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

# set input shape
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# create rpn model here
# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
# rpn outputs regression and cls
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

model_rpn = Model(img_input, rpn[:2])

#load weights from pretrain
try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
#	model_classifier.load_weights(C.base_net_weights, by_name=True)
	print("loaded weights!")
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# compile model
optimizer = Adam(lr=1e-5, clipnorm=0.001)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_rpn.summary()

# write training misc here
epoch_length = 100
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True


# start acutual training here
#X, Y, img_data = next(data_gen_train)
#
##loss_rpn = model_rpn.train_on_batch(X, Y)
#P_rpn = model_rpn.predict_on_batch(X)

# you should enable NMS when you visualize your results.
# NMS will filter out redundant predictions rpn gives, and will only leave the "best" predictions.
# P_rpn = model_rpn.predict_on_batch(image)
# R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
# X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
# this will output the binding box axis. [x1,x2,y1,y2].

Callbacks=keras.callbacks.ModelCheckpoint("./models/rpn/rpn."+options.network+".weights.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=4)
callback=[Callbacks]
if len(val_imgs) == 0:
    # assuming you don't have validation data
    history = model_rpn.fit_generator(data_gen_train,
                    epochs=options.num_epochs, steps_per_epoch = options.epoch_length, callbacks=callback)
    loss_history = history.history["loss"]
else:
    history = model_rpn.fit_generator(data_gen_train,
                    epochs=options.num_epochs, validation_data=data_gen_val,
                    steps_per_epoch=options.epoch_length, callbacks=callback, validation_steps=100)
    loss_history = history.history["val_loss"]

import numpy
numpy_loss_history = numpy.array(loss_history)
numpy.savetxt(options.network+"_rpn_loss_history.txt", numpy_loss_history, delimiter=",")
