# Definition of global hyperparameters and file names
SOURCE_FILE="../data/cityscapes/source_cityscapes.txt"
TARGET_FILE="../data/KITTI/target_kitti.txt"

PARSER="general"
BACKBONE="vgg"
BACK_WEIGHTS="models/phase1/vgg_imagenet_weights.hdf5"
PRETRAINED_WEIGHTS="models/phase1/pascal_voc_2012.hdf5"
N_ROIS=32
PHASE1_CONFIG="phase1_config.pickle"
PHASE1_WEIGHTS="phase1_frcnn.hdf5"

DETECTIONS_DIR="detections"
IMG_DETS_DIR="img_dets"
TARGET_PHASE1_DETECTIONS="target_kitti_dets.txt"

PHASE2_MODEL_INFO="phase2_city_kitti.json"
PHASE2_MODEL_WEIGHTS="phase2_city_kitti.hdf5"
PHASE2_MODEL_TYPE=1

PHASE3_CONFIG="phase3_config.pickle"
PHASE3_WEIGHTS="phase3_city_kitti_weights.hdf5"

# Train phase 1: Original F-RCNN using fine tuning only on the source set
python train_phase1_frcnn.py --network "$BACKBONE" --path "$SOURCE_FILE" --num_epoch 55 --elen 200 \
                             --parser "$PARSER" --input_weight_path "$BACK_WEIGHTS" \
                             --hf --lr 1e-5 --opt SGD --num_rois "$N_ROIS" \
                             --output_weight_path "$PHASE1_WEIGHTS" --config_filename "$PHASE1_CONFIG" \
            		             --load "$PRETRAINED_WEIGHTS"

# Run Phase 1 F-RCNN on target set to get cropped images for phase 2 and save mAP on target set for future comparison
python measure_map.py --path "$TARGET_FILE" --file_type train \
                      --model_path "models/phase1/$BACKBONE\_$PHASE1_WEIGHTS" --parser "$PARSER" --num_rois "$N_ROIS" \
		                  --config_filename "$PHASE1_CONFIG"  \
		                  --dets_flag 1 --dets_dir "$DETECTIONS_DIR" --dets_file "$TARGET_PHASE1_DETECTIONS" \
		                  --img_dets_dir "$IMG_DETS_DIR"

# Train phase 2: Image classifier using the cropped images from the previous step
python train_phase2_classifier.py --source_path "$SOURCE_FILE" --target_path "$DETECTIONS_DIR\/$TARGET_PHASE1_DETECTIONS" \
                                  --original_detector_path "$SOURCE_FILE" --save_dir models/phase2 \
                                  --model_architecture "$PHASE2_MODEL_INFO" --model_weights "$PHASE2_MODEL_WEIGHTS" \
                                  --config_filename "$PHASE1_CONFIG" --num_epochs 60 --e_length 30 --val_size 100 \
                                  --reg_param 0.5 --sup_lr 3e-4 --model_type "$PHASE2_MODEL_TYPE" \
                                  --alpha_init 100.0 --alpha_final 0.5 --hard_constraints --recompute_alpha

# Train phase 3: Robust F-RCNN using all the elements from the previous steps
python train_phase3_frcnn.py --source_path "$SOURCE_FILE" --target_path "$TARGET_FILE" \
                             --parser "$PARSER" --num_rois "$N_ROIS" --num_epochs 55 --elen 50 --opt not_SGD --lr 1e-3 \
                             --phase1_config_file "$PHASE1_CONFIG" --phase1_weights "models/phase1/$BACKBONE\_$PHASE1_WEIGHTS" \
                             --img_json "models/phase2/$PHASE2_MODEL_INFO" --img_weights "models/phase2/$PHASE2_MODEL_WEIGHTS" \
                             --output_config_file "$PHASE3_CONFIG" --output_weight_path "models/phase3/$PHASE3_WEIGHTS" \
                             --alpha_init 100 --alpha_final 0.5 --hard_constraints --recompute_alpha


# Measure the mAP on the testing set of the target dataset with the phase 3 model - optionally save the detections
python measure_map.py --path "$TARGET_FILE" --file_type test \
                      --model_path "models/phase3/$PHASE3_WEIGHTS" --parser "$PARSER" --num_rois "$N_ROIS" \
		                  --config_filename "$PHASE3_CONFIG"  \
		                  --dets_flag 0 --dets_dir "$DETECTIONS_DIR" \
		                  --img_dets_dir "$IMG_DETS_DIR"
		                  # Add the flag --save_imgs or -s to save the images with their bb detections in --img_dets_dir

# To measure the final mAP on the different classes of the source and target domain it's only necessary to run
# the previous script by changing the --path and the --file_type parameters
