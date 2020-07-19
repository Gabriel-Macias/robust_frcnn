# This script can take approximately 8 hours to complete
python train_phase1_frcnn.py --network vgg --path ../data/cityscapes/source_cityscapes.txt --num_epoch 45 --elen 1000 \
                             --parser general --input_weight_path models/phase1/pretrained_vgg16.hdf5 \
                             --hf --lr --opt SGD 1e-5 --num_rois 32  \
		                         --output_weight_path phase1_frcnn.hdf5 --config_filename phase1_config.pickle
