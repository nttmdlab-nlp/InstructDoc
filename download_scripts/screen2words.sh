cd $DATASET_DIR

echo "Donwloading Screen2Words dataset..."
git clone https://github.com/google-research-datasets/screen2words.git
cd screen2words
wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
tar xvf unique_uis.tar.gz && rm unique_uis.tar.gz
