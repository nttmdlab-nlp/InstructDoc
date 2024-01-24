import json
import os
import random
import argparse
import csv

from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
from collections import defaultdict

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'hwsquad'
        self.split = ['train', 'val', 'test']
    
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            filename = os.path.join(self.data_dir, f"HW-SQuAD_{split}_1.0.json")
            with open(filename, "r") as f:
                annotations = json.load(f)

            target_format = []
            for ann in tqdm(annotations["data"]):
                qas = ann["qas"]
                image_path = ann["document_image"]["document_image"]
                h, w = ann["document_image"]["image_height"], ann["document_image"]["image_width"]

                words = []
                bboxes = []
                for item in ann["document_image"]["gold_standard_transcription"]:
                    word = item["text"]
                    words.append(word)
                    bbox = [item["xmin"], item["ymin"], item["xmax"], item["ymax"]]
                    bbox = normalize_bbox(bbox, w, h)
                    bboxes.append(bbox)
                
                for qa in qas:
                    question = qa["question"]
                    start_index, end_index = qa["answers"][0]["answer_start_word_no"], qa["answers"][0]["answer_end_word_no"]+1    
                    answer = words[start_index:end_index]
                    answer = " ".join(answer)

                    instruction = random.choice(instructions)        
                    instruction = instruction.replace('<key>', question)
                    ocr = ' '.join(words)

                    file_name = os.path.abspath(image_path)
                    target_format.append({
                        "image": file_name,
                        "ocr": ocr,
                        "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': answer},
                        ],
                    })

            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/HW-SQuAD/HW-SQuAD_annotations', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/hwsquad', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()