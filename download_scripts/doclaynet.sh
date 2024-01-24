cd $DATASET_DIR

echo "Donwloading DocLaynet dataset..."
mkdir doclaynet
cd doclaynet
wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip
unzip DocLayNet_core.zip && rm DocLayNet_core.zip
unzip DocLayNet_extra.zip && rm DocLayNet_extra.zip
