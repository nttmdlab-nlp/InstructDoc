import json
import os
import random
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
        self.dataset_name = 'visualmrc'
        self.split = ['train', 'dev', 'test']

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            file_name = os.path.join(self.data_dir, f'data/{split}.jsonl')
            with open(file_name, 'r') as f:
                data = f.readlines()
            target_format = []
            for d in tqdm(data):
                d = json.loads(d)
                file_name = os.path.join(self.data_dir, d['image_filename'])
                file_name = os.path.abspath(file_name)
                image = Image.open(file_name)
                w, h = image.size

                words = []
                bboxes = []
                for bbox in d['bounding_boxes']:
                    if 'ocr_info' in bbox:
                        for ocr in bbox['ocr_info']:
                            word = ocr['word']
                            bbox = ocr['bbox']
                            bbox = [bbox['x'], bbox['y'], bbox['x']+bbox['width'], bbox['y']+bbox['height']]
                            bbox = normalize_bbox(bbox, w, h)
                            bboxes.append(bbox)
                            words.append(word)

                ocr = " ".join(words)
                for qa in d['qa_data']:
                    question = qa['question']['text']
                    value = qa['answer']['text']
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', question)

                    target_format.append({
                        "image": file_name,
                        "ocr": ocr, 
                        "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'instruction': instruction},
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
    parser.add_argument('--input_data_dir', default='raw_datasets/VisualMRC_official', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/visualmrc', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()