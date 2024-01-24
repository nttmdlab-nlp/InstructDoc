import json
import os
import random
import argparse

from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import load_instructions
from google_vision_ocr import Google_OCR

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.ocr_dir = os.path.join(args.input_data_dir, 'ocrs')
        self.image_dir = os.path.join(args.input_data_dir, 'images')
        self.dataset_name = 'rvlcdip_io'
        self.google_ocr = Google_OCR(args.api_key)
        self.split = ['train', 'val', 'test']

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            ocrs = []
            with open(os.path.join(self.data_dir, f'labels/{split}.txt'), 'r') as f:
                labels = f.read().splitlines()
            for label in labels:
                filename, label = label.split(' ')
                value = self.class_dict[label]
                image_path = os.path.join(self.image_dir, filename)
                ocr_path = os.path.join(self.ocr_dir, f'{file_name.replace(".tif", ".json")}')
                try:
                    img = Image.open(image_path)
                    img_w, img_h = img.size
                    if not os.path.exists(ocr_path):
                        items = self.google_ocr.recognize_image(img)
                        if items == "error":
                            print('OCR error: ', image_path)
                            continue
                        os.makedirs(os.dirname(ocr_path), exist_ok=True)
                        with open(ocr_path, 'w') as f:
                            json.dump(items, f)
                    else:
                        with open(ocr_path, 'r') as f:
                            items = json.load(f)
                    words, bboxes = self.google_ocr.extract_info(items, img_w, img_h)
                except:
                    words, bboxes = [], []

                ocr = ' '.join(words)
                ocrs.append((ocr, bboxes))

            for label in labels:
                instruction = random.choice(instructions)
                if random.random() > 0.5:
                    ocr, bboxes = random.choice(ocrs)
                    value = 'no'
                else:
                    value = 'yes'
                
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
    parser.add_argument('--input_data_dir', default='raw_datasets/rvlcdip', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/rvlcdip_io', type=str)
    parser.add_argument('--api_key', default='API_KEY', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()