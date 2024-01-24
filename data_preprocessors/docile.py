import json
import os
import random
from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import sort_coordinate, load_instructions, normalize_bbox
import argparse
from collections import defaultdict

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'docile'
        self.ann_dir = os.path.join(args.input_data_dir, f'annotations')
        self.img_dir = os.path.join(args.input_data_dir, f'images')
        self.ocr_dir = os.path.join(args.input_data_dir, f'ocr')
        self.split = ['train', 'val']

    def extract_ocr_info(self, ocr_data):
        tokens = []
        bboxes = []
        for page in ocr_data['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        left_top, right_bottom = word['geometry']
                        bbox = normalize_bbox([left_top[0], left_top[1], right_bottom[0], right_bottom[1]])
                        bboxes.append(bbox)
                        tokens.append(word['value'])
        return tokens, bboxes

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            file_name = os.path.join(self.data_dir, f'{split}.json')
            with open(file_name, 'r') as f:
                ann_filenames = json.load(f)

            target_format = []
            for id, file in enumerate(tqdm(ann_filenames)):
                image_path = os.path.join(self.img_dir, file + '0001-1.jpg')
                with open(os.path.join(self.ocr_dir, f'{file}.json'), 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                with open(os.path.join(self.ann_dir, f'{file}.json'), 'r', encoding='utf-8') as f:
                    d = json.load(f)

                items = []
                for item in d["field_extractions"]:
                    if item["page"] == 0:                
                        text, label = item["text"], item["fieldtype"]
                        bbox = item["bbox"]
                        items.append((text, label, bbox))
                if len(items) == 0:
                    continue
                items = sort_coordinate(items)

                labels = {}
                for item in items:
                    tokens, label, bbox = item
                    labels[tokens] = label

                tokens, bboxes = self.extract_ocr_info(ocr_data)
                ocr = ' '.join(tokens)

                for key in labels:
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', key)
                    value = labels[key]

                    file_name = os.path.abspath(image_path)
                    target_format.append({
                        "image": file_name,
                        "ocr": ocr,
                        "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': value},
                        ],
                    })

            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/docile/data/docile', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/docile', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()