import json
import os
import random
import argparse
import csv

from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'tatdqa'
        self.split = ['train', 'dev', 'test']
    
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            file_name = os.path.join(self.data_dir, f'tatdqa_dataset_{split}.json')
            with open(file_name, 'r') as f:
                data = json.load(f)
            for d in tqdm(data):
                uid = d['doc']['uid']
                page_num = d['doc']['page']
                image_path = f'{split}/{uid}_{page_num}.png'
                ocr_file_name = os.path.join(self.data_dir, f'{split}/{uid}.json')
                with open(ocr_file_name, 'r') as f:
                    ocrs = json.load(f)

                text = []
                bboxes = []
                _, _, w, h = ocrs['pages'][page_num-1]['bbox']
                for page in ocrs['pages']:
                    for block in page['blocks']:
                        text.append(block['text'])
                        for bbox in block['words']['bbox_list']:
                            bbox = normalize_bbox(bbox, w, h)
                            bboxes.append(bbox)

                for qa in d['questions']:
                    question =qa['question']
                    if 'answer' in qa:
                        answer = qa['answer']
                        if type(qa['answer']) == list:
                            if len(qa['answer']) > 1:
                                answer = ', '.join(answer)
                            else:
                                answer = answer[0]
                    else:
                        answer = ""

                    instruction = random.choice(instructions)        
                    instruction = instruction.replace('<key>', question)
                    ocr = ' '.join(text)

                    file_name = os.path.abspath(image_path)
                    target_format.append({
                        "image": file_name,
                        "ocr": ocr,
                        "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': answer},
                        ],
                    })

            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/TAT-DQA', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/tatdqa', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()