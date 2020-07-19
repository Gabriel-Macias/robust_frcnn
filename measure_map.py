import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.vgg as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score


def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False
        bbox['difficult'] = 1

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    return T, P


def format_img(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis = 0)
    return img, fx, fy, f


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest = "data_path", help = "Path to data txt file.")
parser.add_option("-t", "--file_type", dest = "file_type", help = "File type of test data. 'train' (default) or 'test'.", default = "train")
parser.add_option("--model_path", dest = "weights_path", help = "Path to model weights.")
parser.add_option("-o", "--parser", dest = "parser", help = "Parser to use. One of  pascal_voc or general", default = "general")
parser.add_option("-n", "--num_rois", dest = "num_rois", help = "Number of ROIs per iteration. Higher means more memory use.", default = 32)
parser.add_option("--config_filename", dest = "config_filename", help = "Path to config file (generated when training phase 1).", default = "config.pickle")
parser.add_option("-d", "--dets_dir", dest = "dets_dir", help = "Path to dir to save the txt file with the target detections.", default = "detections")
parser.add_option("-f", "--dets_file", dest = "dets_file", help = "Name of the file that will be created with the detections.", default = "target_detections.txt")
parser.add_option("--dets_flag", dest = "dets_flag", help = "Flag to 1: store the detections or 0: not (default).", default = 0, type = int)
parser.add_option("-i", "--img_dets_dir", dest = "img_dets_dir", help = "Path to dir inside dets_dir to save the images with bb detections.", default = "img_dets")
parser.add_option("-s", "--save_imgs", dest = "save_imgs", help = "Flag used to save imgs with detections (True) or not (False).", action = "store_true", default=False)

(options, args) = parser.parse_args()

# If data path is not given
if not options.data_path:
    parser.error('Error: path to test data must be specified. Pass --path to command line')

if not options.weights_path:
    parser.error('Error: path to model weights must be specified. Pass --model_path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'general':
    from keras_frcnn.general_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# Turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

model_path = options.weights_path

if not os.path.isdir(options.dets_dir):
    os.mkdir(options.dets_dir)

imgs_dets_dir = os.path.join(options.dets_dir, options.img_dets_dir)
if not os.path.isdir(imgs_dets_dir):
    os.mkdir(imgs_dets_dir)

if os.path.isfile(os.path.join(options.dets_dir, options.dets_file)) and options.dets_flag != 0:
    os.remove(os.path.join(options.dets_dir, options.dets_file))

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
C.num_rois = int(options.num_rois)

if options.save_imgs:
    class_to_color = {inv_class_mapping[v]: np.random.randint(0, 255, 3) for v in inv_class_mapping}

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (512, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 512)

img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (C.num_rois, 4))
feature_map_input = Input(shape = input_shape_features)

# Define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable = False)

# Define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes = len(class_mapping), trainable = False)

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)

C.model_path = model_path

model_rpn.load_weights(C.model_path, by_name = True)
model_classifier.load_weights(C.model_path, by_name = True)

model_rpn.compile(optimizer = 'sgd', loss = 'mse')
model_classifier.compile(optimizer = 'sgd', loss = 'mse')

all_imgs, _, _ = get_data(options.data_path)
test_imgs = [s for s in all_imgs if s['imageset'] == options.file_type]

if len(test_imgs) == 0:
    raise NameError('No images to process. Verify the path to testing images file and file_type parameter.')

print("Number of images to process: {}".format(len(test_imgs)))
print("============ Starting Detections ============")

T = {}
P = {}
st = time.time()
for idx, img_data in enumerate(test_imgs):

    if idx % 100 == 0 and idx > 0:
        print('================== {}/{} =================='.format(idx, len(test_imgs)))

    filepath = img_data['filepath']

    img = cv2.imread(filepath)

    X, fx, fy, f = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh = 0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # Apply the spatial pyramid pooling to the proposed regions
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

        [P_cls, P_regr] = model_classifier.predict([F, ROIs])

        # Count the number of positive and negative detections - Save at most 2 background detections per foreground (positive) detections
        idx_neg = np.where(P_cls[0].argmax(1) == (P_cls.shape[2] - 1))[0]
        n_pos = P_cls.shape[1] - idx_neg.shape[0]
        n_neg = min([2 * n_pos, idx_neg.shape[0]])
        if n_neg > 0:
            idx_neg = np.random.choice(idx_neg, n_neg, replace = False)

        for ii in range(P_cls.shape[1]):

            if options.dets_flag == 1 and ii in idx_neg:
                pass
            elif np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = inv_class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
                all_probs[cls_name] = []

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
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
            all_probs[cls_name].append(P_cls[0, ii, :])

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs, chosen_idx = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.1)
        new_all_probs = np.array(all_probs[key])[chosen_idx, :]

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}

            if key != 'bg':
                all_dets.append(det)

            if options.dets_flag == 1:
                proba_str = "|".join(new_all_probs[jk, :].astype("str"))

                (real_x1, real_y1, real_x2, real_y2) = data_generators.get_real_coordinates(f, x1, y1, x2, y2)
                with open(os.path.join(options.dets_dir, options.dets_file), "a") as out_f:
                    out_f.write("{},{},{},{},{},{},{}\n".format(filepath, real_x1, real_y1, real_x2, real_y2, key, proba_str))

            if options.save_imgs:
                (real_x1, real_y1, real_x2, real_y2) = data_generators.get_real_coordinates(f, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    if options.save_imgs:
        cv2.imwrite(os.path.join(imgs_dets_dir, "{}.png".format(idx)), img)

    t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
    all_aps = []

    for key in T.keys():
        try:
            ap = average_precision_score(T[key], P[key])
        except:
            continue
        if idx % 100 == 0 and idx > 0:
            print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)

    if idx % 100 == 0 and idx > 0:
        if len(all_aps) > 0:
            print('mAP = {}'.format(np.nanmean(np.array(all_aps))))
        else:
            print("No detections so far at iteration {}".format(idx))
        print('Elapsed time = {}'.format(time.time() - st))
        st = time.time()

print("======================== FINISHED ========================")
print('mAP = {}'.format(np.nanmean(np.array(all_aps))))
for key in T.keys():
    ap = average_precision_score(T[key], P[key])
    print('{} AP: {}'.format(key, ap))
