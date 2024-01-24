cd $DATASET_DIR

echo "Donwloading IconQA dataset..."
mkdir iconqa
cd iconqa
wget https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip
unzip iconqa_data.zip && rm iconqa_data.zip
