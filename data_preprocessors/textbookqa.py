import json
import os
import random
import glob
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, sort_coordinate, load_instructions
from transformers import BertTokenizer
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'textbookqa'
        self.split = ['train', 'val', 'test']

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for split in self.split:
            target_format = []
            ann_filename = f'{split}/tqa_v1_{split}.json' if split != 'test' else f'{split}/tqa_v2_{split}.json'
            ann_filename = os.path.join(self.data_dir, ann_filename)
            with open(ann_filename, 'r') as f:
                anns = json.load(f)
            for ann in tqdm(anns):
                questions = ann['questions']
                diagram_questions = questions['diagramQuestions']
                if len(diagram_questions) == 0:
                    continue

                diagram_annotations = ann['diagramAnnotations']

                for global_id, data in diagram_questions.items():
                    options = []
                    for option_id, choice in data['answerChoices'].items():
                        choice = choice['processedText']
                        options.append(choice)
                    question = data['beingAsked']['processedText']
                    value = data['correctAnswer']['rawText']
                    image_path = data['imagePath']
                    image_name = data['imageName']
                    image_path = os.path.join(self.data_dir, f'{split}/{image_path}')
                    if image_name in diagram_annotations:
                        annotation = diagram_annotations[image_name]
                        bboxes = []
                        ocr = []
                        for item in annotation:
                            text, bbox = item["text"], item["rectangle"]
                            try:
                                bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
                            except:
                                continue
                            if len(text) > 0:
                                bboxes.append(bbox)
                                ocr.append(text)
                        ocr = " ".join(ocr)
                    else:
                        ocr = ""
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', question).replace('<options>', str(options))

                    file_name = os.path.abspath(image_path)
                    target_format.append({
                        "image": file_name,
                        "ocr": ocr,
                        "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': f"{value}"},
                        ],
                    })
            
            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/textbookqa', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/textbookqa', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()