cd $DATASET_DIR

echo "Donwloading AI2D dataset..."
mkdir ai2d
cd ai2d
wget https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip
wget https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv
unzip ai2d-all.zip && rm ai2d-all.zip
