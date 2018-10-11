#!/bin/bash
if [ ! -e data/CLEVR_v1.0 ]; then
    wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0_no_images.zip
    unzip CLEVR_v1.0_no_images.zip -d data/
    rm -rf CLEVR_v1.0_no_images.zip
fi
if [ ! -e data/processed ]; then
    echo "processing images..."
    mkdir data/processed
    pip2 install --user scipy h5py nltk
    python2 src/preprocess.py --input_train_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
	   --input_dev_json data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
	   --input_test_json data/CLEVR_v1.0/questions/CLEVR_test_questions.json \
	   --outputdir data/processed
    echo " done"
fi
