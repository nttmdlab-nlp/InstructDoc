import json
import os
import random

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
        self.dataset_name = 'cord'
        self.split = ['train', 'dev', 'test']

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            ann_dir = os.path.join(self.data_dir, f'{split}/json')
            img_dir = os.path.join(self.data_dir, f'{split}/image')
            for file in tqdm(sorted(os.listdir(ann_dir))):
                file_path = os.path.join(ann_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_path = os.path.join(img_dir, file)
                image_path = image_path.replace('.json', '.png')
                image = Image.open(image_path)
                w, h = image.size

                items = []
                labels = {}
                for item in data["valid_line"]:
                    words, label = item["words"], item["category"]
                    words = [w for w in words if w["text"].strip() != ""]
                    if len(words) == 0:
                        continue
                    text = " ".join([word["text"] for word in words])
                    bbox = [words[0]["quad"]["x1"], words[0]["quad"]["y1"], words[-1]["quad"]["x3"], words[-1]["quad"]["y3"]]
                    bbox = normalize_bbox(bbox, w, h)
                    items.append((text, label, bbox))

                items = sort_coordinate(items)
                ocr = []
                bboxes = []
                for item in items:
                    words, label, bbox = item
                    labels[words] = label
                    ocr.append(words)
                    bbox = [bbox] * len(words.split())
                    bboxes += bbox
                ocr = ' '.join(ocr)

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

            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/cord/CORD', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/cord', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()