import json
import os
import random
import glob
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'iconqa'
        self.split = ['train', 'val']

    def create_data(self):
        for split in self.split:
            for answer_style in ['fill_in_blank', 'choose_txt']:
                target_format = []
                dataset_name = f'{self.dataset_name}_{answer_style}'
                instructions = load_instructions(self.instruction_path)[dataset_name]

                data_dir = os.path.join(self.data_dir, f'{split}/{answer_style}/*')
                for file_path in glob.glob(data_dir):
                    data_path = os.path.join(file_path, 'data.json')
                    image_path = os.path.join(file_path, 'image.png')
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    question = data['question']
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', question)
                    if answer_style == 'fill_in_blank':
                        value = data['answer']
                    else:
                        options = data['choices']
                        answer_index = data['answer']
                        value = str(options[answer_index])
                        instruction = instruction.replace('<options>', options)

                    file_name = os.path.abspath(image_path)
                    target_format.append({
                        "image": file_name,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': f"{value}"},
                        ],
                    })
            
                out_data_dir = f'{self.out_data_dir}_{answer_style}'
                out_filepath = os.path.join(out_data_dir, f'{split}.json')        
                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

                print(f'{split}: {len(target_format)}')
                with open(out_filepath, "w") as f:
                    json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/iconqa/iconqa_data', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/iconqa', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()