cd $DATASET_DIR

echo "Donwloading AI2D dataset..."
mkdir ai2d
cd ai2d
wget https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip
unzip ai2d-all.zip && rm ai2d-all.zip
