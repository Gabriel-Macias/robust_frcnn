# Fine tuning training times can be much shorter
python train_phase1_frcnn.py --network vgg --path ../data/cityscapes/source_cityscapes.txt --num_epoch 55 --elen 200 \
                             --parser general --input_weight_path models/phase1/pretrained_vgg16.hdf5 \
                             --hf --lr 1e-5 --opt SGD --num_rois 32 \
                             --output_weight_path phase1_frcnn.hdf5 --config_filename phase1_fine_tuning.pickle \
            		             --load models/phase1/pascal_voc_2012.hdf5
