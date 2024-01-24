import json
import os
import random
import argparse

from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
from collections import defaultdict
from google_vision_ocr import Google_OCR

class InstructData:
    def __init__(self, args):
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.ocr_dir = os.path.join(args.input_data_dir, 'ocrs')
        self.image_dir = os.path.join(args.input_data_dir, 'images')
        self.google_ocr = Google_OCR(args.api_key)
        os.makedirs(self.ocr_dir, exist_ok=True)

    def create_data(self):
        file_name = os.path.join(self.data_dir, 'llava_instruct_150k_llavar_20k.json')
        with open(file_name, 'r') as f:
            data = json.load(f)
        target_format = []
        for d in data:
            image_name = d["image"]
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(image_path):
                continue

            ocr_path = os.path.join(self.ocr_dir, f"{image_name.replace('.jpg', '.json')}")
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
                words, bboxes = self.google_ocr.extract_info(items, img_w, img_h)
            except:
                words, bboxes = [], []

            ocr = ' '.join(words)
            file_name = os.path.abspath(image_path)
            d["image"] = file_name
            d["ocr"] = ocr
            d["bboxes"] = bboxes
            target_format.append(d)

        out_filepath = os.path.join(self.out_data_dir, 'train.json')        
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

        print(f'train: {len(target_format)}')
        with open(out_filepath, "w") as f:
            json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/llavar', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/llavar', type=str)
    parser.add_argument('--api_key', default='API_KEY', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()