cd $DATASET_DIR

echo "Donwloading WebSRC dataset..."
mkdir websrc
cd websrc
wget https://websrc-data.s3.amazonaws.com/release.zip
unzip release.zip && rm release.zip
