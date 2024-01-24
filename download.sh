#!/bin/bash
export DATASET_DIR=raw_datasets

mkdir raw_datasets  

sh ./download_scripts/llavar.sh
<< COMMENT
sh ./download_scripts/due.sh
sh ./download_scripts/websrc.sh
sh ./download_scripts/funsd.sh
sh ./download_scripts/iconqa.sh
sh ./download_scripts/textbookqa.sh
sh ./download_scripts/screen2words.shsh 
sh ./download_scripts/doclaynet.sh
sh ./download_scripts/ai2d.sh
sh ./download_scripts/wildreceipt.sh

# font file for rendering text in AI2D dataset
wget https://huggingface.co/Team-PIXEL/pixel-base-finetuned-masakhaner-swa/resolve/main/GoNotoCurrent.ttf
COMMENT