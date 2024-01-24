import json
import os
import random
import glob
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, sort_coordinate, load_instructions
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'scienceqa'

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        train, val, test = [],[],[]
        target_format = []
        ann_filename = os.path.join(self.data_dir, 'data/scienceqa/problems.json')
        with open(ann_filename, 'r') as f:
            anns = json.load(f)
        for questionId, ann in tqdm(anns.items()):
            question = ann['question']
            choices = ann['choices']
            value = choices[ann['answer']]
            split = ann['split']
            image_name = ann['image']
            if str(image_name) == 'null':
                continue

            image_path = os.path.join(self.data_dir, split, questionId, image_name)
            instruction = random.choice(instructions)
            instruction = instruction.replace('<key>', question).replace('<options>', str(choices))

            file_name = os.path.abspath(image_path)
            sample = {
                "image": file_name,
                "conversations": [
                    {'from': 'human', 'value': instruction},
                    {'from': 'gpt', 'value': f"{value}"},
                ],
            }
            if split == 'train':
                train.append(sample)
            elif split == 'val':
                val.append(sample)
            elif split == 'train':
                test.append(sample)
        
        for split, target_format in [('train', train), ('val', val), ('test', test)]:
            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/scienceqa', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/scienceqa', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()