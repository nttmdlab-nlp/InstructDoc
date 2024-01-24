import json
import os
import random
import argparse
import csv

from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
from collections import defaultdict
from google_vision_ocr import Google_OCR

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.ocr_dir = os.path.join(args.input_data_dir, 'ocrs')
        self.dataset_name = 'websrc'
        self.google_ocr = Google_OCR(args.api_key)
        self.split = ['train', 'dev']
        os.makedirs(self.ocr_dir, exist_ok=True)
    
    def load_split_info(self):
        file_name = os.path.join(self.data_dir, 'dataset_split.csv')
        with open(file_name) as f:
            reader = csv.reader(f)
            split_info = defaultdict(list)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                number = '0' + row[1] if int(row[1]) < 10 else  row[1]
                split = row[3]
                data_path = os.path.join(self.data_dir, f'{row[0]}/{number}/dataset.csv')
                split_info[split].append(data_path)
        return split_info
        
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        split_info = self.load_split_info()
        for split in self.split:
            target_format = []
            for data_path in tqdm(split_info[split]):
                with open(data_path) as f:
                    data_dir = os.path.dirname(data_path)
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i == 0:
                            for index, element in enumerate(row):
                                if 'question' == element:
                                    question_index = index
                                elif 'id' == element:
                                    id_index = index
                                elif 'answer' == element:
                                    answer_index = index
                            continue   
                        questionId = row[id_index]
                        image_path = os.path.join(data_dir, f'processed_data/{questionId[2:9]}.png')
                        img = Image.open(image_path)
                        img_w, img_h = img.size

                        ocr_path = os.path.join(self.ocr_dir, f'{questionId[2:9]}.json')
                        try:
                            if not os.path.exists(ocr_path):
                                items = self.google_ocr.recognize_image(img)
                                if items == "error":
                                    print('OCR error: ', image_path)
                                    continue
                                with open(ocr_path, 'w') as f:
                                    json.dump(items, f)
                            else:
                                with open(ocr_path, 'r') as f:
                                    items = json.load(f)
                            words, bboxes = self.google_ocr.extract_info(items, img_w, img_h)
                        except:
                            words, bboxes = [], []

                        question = row[question_index]
                        instruction = random.choice(instructions)        
                        instruction = instruction.replace('<key>', question)
                        ocr = ' '.join(words)
                        value = row[answer_index]

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
    parser.add_argument('--input_data_dir', default='raw_datasets/websrc', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/websrc', type=str)
    parser.add_argument('--ocr_dir', default='raw_datasets/websrc/ocrs', type=str)
    parser.add_argument('--api_key', default='API_KEY', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()