# 55 epochs and 50 elen -> 8 hrs
python train_phase3_frcnn.py --source_path ../data/cityscapes/source_cityscapes.txt --target_path ../data/KITTI/target_kitti.txt \
                             --parser general --num_rois 32 --num_epochs 55 --elen 50 --opt not_SGD --lr 1e-3 \
                             --phase1_config_file phase1_tuning.pickle --phase1_weights models/phase1/vgg_cityscapes.hdf5 \
                             --img_json models/phase2/phase2_model.json --img_weights models/phase2/phase2_weights.hdf5 \
                             --output_config_file phase3_config.pickle --output_weight_path models/phase3/phase3_weights.hdf5 \
                             --alpha_init 100 --alpha_final 0.5 --hard_constraints --recompute_alpha
