cd $DATASET_DIR

echo "Donwloading TextbookQA dataset..."
mkdir textbookqa
cd textbookqa
wget https://ai2-public-datasets.s3.amazonaws.com/tqa/tqa_train_val_test.zip
unzip tqa_train_val_test.zip && rm tqa_train_val_test.zip
