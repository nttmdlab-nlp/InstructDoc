cd $DATASET_DIR

echo "Donwloading LLaVAR dataset..."
mkdir llavar
cd llavar
wget https://huggingface.co/datasets/SALT-NLP/LLaVAR/resolve/main/llava_instruct_150k_llavar_20k.json
mkdir images
cd images
wget https://huggingface.co/datasets/SALT-NLP/LLaVAR/resolve/main/finetune.zip
unzip finetune.zip && rm finetune.zip
