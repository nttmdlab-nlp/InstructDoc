import json
import os
import random
import argparse
import glob

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
        self.dataset_name = 'slidevqa'
        self.google_ocr = Google_OCR(args.api_key)
        self.split = ['train', 'val', 'test']
        os.makedirs(self.ocr_dir, exist_ok=True)
            
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            file_name =  os.path.join(self.data_dir, 'annotations/qa', f'{split}.jsonl')
            with open(file_name, 'r') as f:
                data = f.read().splitlines()
            for d in tqdm(data):
                question = d['question']
                deck_name = d['deck_name']
                value = d['answer']
                image_paths = []
                text_sequences = []
                bboxes = []
                for image_path in glob.glob(os.path.join(self.image_dir, deck_name, f'slide_*_1024.jpg')):
                    image_path = os.path.abspath(image_path)
                    image_name = os.path.basename(image_path)                
                    image_paths.append(image_path)
                    ocr_path = os.path.join(self.ocr_dir, f'{deck_name}_{image_name.replace(".jpg", ".json")}')
                    try:
                        img = Image.open(image_path)
                        img_w, img_h = img.size
                        if not os.path.exists(ocr_path):
                            items = self.google_ocr.recognize_image(img)
                            if items == 'error':
                                print('OCR error: ', image_path)
                                continue
                            with open(ocr_path, 'w') as f:
                                json.dump(items, f)
                        else:
                            with open(ocr_path, 'r') as f:
                                items = json.load(f)
                        words, page_bboxes = self.google_ocr.extract_info(items, img_w, img_h)
                    except:
                        words, page_bboxes = [], []                        
                    text_sequences.append(' '.join(words))
                    bboxes.append(page_bboxes)
                
                instruction = random.choice(instructions)
                instruction = instruction.replace('<key>', question)

                file_name = os.path.abspath(image_path)
                target_format.append({
                    "image_list": image_paths,
                    "ocr_list": text_sequences, 
                    "bboxes_list": bboxes, 
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
    parser.add_argument('--input_data_dir', default='raw_datasets/slidevqa', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/slidevqa', type=str)
    parser.add_argument('--api_key', type=str, help='google vision api key')
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()