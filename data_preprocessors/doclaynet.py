import json
import os
import random
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, sort_coordinate, load_instructions, convert_wh
from collections import defaultdict
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'doclaynet'
        self.split = ['train', 'val']

    def sort_coordinate(self, bboxes):
        return sorted(bboxes , key=lambda k: [k[1][1], k[1][0]])    

    def create_ocr_data(self, data):
        ocr_info = {}
        for image_info in data['images']:
            file_name = image_info['file_name']
            image_id = image_info['id']
            image_path = os.path.join(self.data_dir,  f'PNG/{file_name}')
            json_path = os.path.join(self.data_dir,  f'JSON/{file_name.replace(".png", ".json")}')
            width, height = image_info['width'], image_info['height']
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            items = []

            for cell in json_data['cells']:
                text = cell['text']
                bbox = cell['bbox']
                bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                bbox = convert_wh(normalize_bbox(bbox, width, height))
                items.append((text, bbox))

            items = self.sort_coordinate(items)
            words = []
            bboxes = []
            for text, bbox in items:
                words.append(text)
                bboxes += bbox
            text_sequence = ' '.join(words)
            ocr_info[image_id] = {'image_path': image_path, 'text_sequence': text_sequence, 'bboxes': bboxes, 'width': width, 'height': height}
            break
        return ocr_info
    
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            with open(os.path.join(self.data_dir, f'COCO/{split}.json'), "r") as f:
                data = json.load(f)
            ocr_info = self.create_ocr_data(data)
            categories = data['categories']

            target_format = []
            annotations = defaultdict(list)
            for ann_info in data['annotations']:
                image_id = ann_info['image_id']
                annotations[image_id].append(ann_info)

            for image_id in tqdm(annotations):
                image_info = ocr_info[image_id]
                image_path = image_info['image_path']
                text_sequence = image_info['text_sequence']
                bboxes = image_info['bboxes']
                width, height = image_info['width'], image_info['height']

                items = []
                for ann in annotations[image_id]:
                    category_id = ann['category_id']
                    category_name = categories[category_id-1]['name']
                    bbox = ann['bbox']
                    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[2], bbox[3]]
                    bbox = normalize_bbox(bbox, width, height)
                    items.append((category_name, bbox))
                items = self.sort_coordinate(items)

                dla = []
                for item in items:
                    category_name, bbox = item
                    dla.append(f'{category_name} {bbox}')
                value = ' '.join(dla)

                instruction = random.choice(instructions)        
                file_name = os.path.abspath(image_path)

                target_format.append({
                    "image": file_name,
                    "ocr": text_sequence,
                    "bboxes": bboxes,
                    "conversations": [
                        {'from': 'human','value': instruction},
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
    parser.add_argument('--input_data_dir', default='raw_datasets/doclaynet', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/doclaynet', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()