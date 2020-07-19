# Script to measure MAP by class and global - optionally saves detections for phase 2 and images with bb detections
python measure_map.py --path ../data/cityscapes/target_kitti.txt --file_type test \
                      --model_path models/phase3/phase3_city_kitti_weights.hdf5 --parser general --num_rois 32 \
		                  --config_filename phase3_tuning.pickle  \
		                  --dets_flag 0 --dets_dir detections/ --dets_file target_kitti_dets.txt \
		                  --img_dets_dir img_dets
		                  # Add the flag --save_imgs or -s to save the images with their detections in --img_dets_dir
