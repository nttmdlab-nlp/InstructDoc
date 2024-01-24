import json
import os
import random
import argparse
import glob

from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.image_dir = os.path.join(args.input_data_dir, 'DUDE_train-val-test_binaries/images')
        self.ocr_dir = os.path.join(args.input_data_dir, 'DUDE_train-val-test_binaries/OCR')
        self.dataset_name = 'dude'
        self.split = ['train', 'val']
            
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        file_name =  os.path.join(self.data_dir, '2023-03-23_DUDE_gt_test_PUBLIC.json')
        with open(file_name, 'r') as f:
            data = json.load()
        train, validation = [],[]
        for d in tqdm(data['data']):
            docid = d['docId']
            question = d['question']
            split = d['data_split']          
            if split in self.split:
                image_paths = []
                pages = len(glob.glob(os.path.join(self.image_dir, split, f'{docid}_*.jpg')))
                for i in range(pages):
                    image_path = os.path.join(self.image_dir, split, f'{docid}_{i}.jpg')
                    image_path = os.path.abspath(image_path)                
                    image_paths.append(image_path)

                ocr_path =os.path.join(self.ocr_dir, f'Azure/{docid}_due.json')
                try:
                    with open(ocr_path, 'r') as f:
                        ocr_info = json.load(f)
                except:
                    continue

                structure_value = ocr_info['structures']['pages']['structure_value']
                image_sizes = ocr_info['structures']['pages']['positions']
                text_sequences = []
                bboxes = []
                for page_split, image_size in zip(structure_value, image_sizes):
                    start = page_split[0]
                    end = page_split[1]
                    page_tokens = ' '.join(ocr_info['tokens'][start:end])
                    page_bboxes = []
                    for bbox in ocr_info['positions'][start:end]:
                        bbox = normalize_bbox(bbox, (image_size[2], image_size[3]))
                        page_bboxes.append(bbox)
                    text_sequences.append(page_tokens)
                    bboxes.append(page_bboxes)
                                
                if len(text_sequences) != len(image_paths):
                    continue

                instruction = random.choice(instructions)
                instruction = instruction.replace('<key>', question)
                if 'answers' in d:
                    value = d['answers'][0]
                    if d['answer_type'] == 'not-answerable':
                        d['answers'] = 'none'
                else:
                    value  = ''

                file_name = os.path.abspath(image_path)
                sample = {
                    "image_list": image_paths,
                    "ocr_list": text_sequences, 
                    "bboxes_list": bboxes, 
                    "conversations": [
                        {'from': 'human', 'value': instruction},
                        {'from': 'gpt', 'value': value},
                    ],
                }

                if split == 'train':
                    train.append(sample)
                elif split == 'val':
                    validation.append(sample)

        for split, target_format in [('train', train), ('validation', validation)]:
            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/dude', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/dude', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()