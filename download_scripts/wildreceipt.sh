cd $DATASET_DIR

echo "Donwloading WildReceipt dataset..."
mkdir wildreceipt
cd wildreceipt
wget https://download.openmmlab.com/mmocr/data/wildreceipt.tar
tar xvf wildreceipt.tar && rm wildreceipt.tar
