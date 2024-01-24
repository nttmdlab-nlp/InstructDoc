import json
import os
import random
from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, sort_coordinate, load_instructions
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'wildreceipt'
        self.split = ['train', 'test']
        self.classes = {}
        for items in open(os.path.join(args.input_data_dir, 'class_list.txt')):
            index, label = items.split()
            self.classes[index] = label

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            with open(os.path.join(self.data_dir, f'{split}.txt')) as f:
                samples = f.readlines()
            for sample in tqdm(samples):
                data = json.loads(sample)
                file_name = data['file_name']
                image_path = os.path.join(self.data_dir, file_name)
                image = Image.open(image_path)
                w, h = image.size

                items = []
                labels = {}
                for item in data["annotations"]:
                    text, label_index = item["text"], item["label"]
                    label = self.classes[str(label_index)]
                    if label_index == 0:
                        continue
                    bbox = item["box"]
                    bbox = [bbox[0], bbox[1], bbox[4], bbox[5]]
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
    parser.add_argument('--input_data_dir', default='raw_datasets/wildreceipt/wildreceipt', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/wildreceipt', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()