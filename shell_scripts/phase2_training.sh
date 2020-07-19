# epochs 25 e_length 1000 -> 3 hrs
python train_phase2_classifier.py --source_path ../data/cityscapes/source_cityscapes.txt --target_path detections/target_kitti_dets.txt \
                                  --original_detector_path ../data/cityscapes/source_cityscapes.txt --save_dir models/phase2 \
                                  --model_architecture phase2_model.json --model_weights phase2_weights.hdf5 \
                                  --config_filename fine_tuning.pickle --num_epochs 60 --e_length 30 --val_size 100 \
                                  --reg_param 0.5 --sup_lr 3e-4 --model_type 1 --alpha_init 100.0 --alpha_final 0.5 \
                                  --hard_constraints --recompute_alpha
