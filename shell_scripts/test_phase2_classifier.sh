# Script to measure the F1-Score of --num_samples cropped images from the --data_path file
python test_phase2_classifier.py --data_path ../data/cityscapes/car_person_cityscapes.txt \
                                 --model_architecture phase2_model.json --model_weights phase2_weights.hdf5 \
                                 --num_samples 100 --config_filename fine_tuning.pickle
