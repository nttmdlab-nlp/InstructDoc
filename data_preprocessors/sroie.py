import json
import os
import random
import cv2
from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'sroie'
        self.split = ['train', 'test']

    def sort_coordinate(self, bboxes):
        return sorted(bboxes , key=lambda k: [k[1][1], k[1][0]])    

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            ann_dir = os.path.join(self.data_dir, f'{split}/entities')
            img_dir = os.path.join(self.data_dir, f'{split}/img')
            for file in tqdm(sorted(os.listdir(ann_dir))):
                file_path = os.path.join(ann_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                image_path = os.path.join(img_dir, file)
                image_path = image_path.replace('.txt', '.jpg')
                image = cv2.imread(image_path)
                h, w, _ = image.shape
                    
                file_path = os.path.join(ann_dir.replace('entities', 'box'), file)
                text_sequence = []
                bboxes = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    items = []
                    for item in f.read().splitlines():
                        bbox = item.split(',')[:8]
                        text = item[len(','.join(bbox))+1:]
                        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[4]), int(bbox[5])]
                        bbox = normalize_bbox(bbox, w, h)
                        items.append((text, bbox))
                items = self.sort_coordinate(items)
                for item in items:
                    words, bbox = item
                    text_sequence.append(words)
                    bbox = [bbox] * len(words.split())
                    bboxes += bbox
                
                ocr = ' '.join(text_sequence)
                for label in labels:
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', labels[label])

                    file_name = os.path.abspath(image_path)
                    target_format.append({
                        "image": file_name,
                        "ocr": ocr,
                        "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': label},
                        ],
                    })

            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/SROIE2019', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/sroie', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()