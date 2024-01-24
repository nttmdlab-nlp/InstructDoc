import json
import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
from transformers import BertTokenizer
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.out_data_dir = args.out_data_dir
        self.question_dir = os.path.join(args.input_data_dir, f'questions')
        self.ann_dir = os.path.join(args.input_data_dir, f'annotations')
        self.img_dir = os.path.join(args.input_data_dir, f'images')
        self.font = ImageFont.truetype(args.font_file, size=40)
        self.dataset_name = 'ai2d'
        self.split = ['train', 'test']

    def sort_coordinate(self, bboxes):
        return sorted(bboxes, key=lambda k: [k[1][1], k[1][0]])    

    def create_data(self):
        train = []
        test = []
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        with open(os.path.join(self.data_dir, 'ai2d_test_ids.csv')) as f:
            test_ids = f.read().splitlines()
        for i, file in enumerate(tqdm(sorted(os.listdir(self.question_dir)))):
            file_path = os.path.join(self.question_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            annotation_path = os.path.join(self.ann_dir, file)
            with open(annotation_path, 'r') as f:
                ann = json.load(f) 

            index = file.replace('.png.json', '')
            split = 'test' if str(index) in test_ids else 'train'

            image_path = os.path.join(self.img_dir, file)
            image_path = image_path.replace('.json', '')
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            for index, text in ann['text'].items():
                replacement_text = text['replacementText']
                bbox = text['rectangle']
                bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
                text = text['value']
                x1, y1, x2, y2 = bbox
                draw.rectangle((x1, y1, x2, y2), outline="lime", width=4)
                draw.text((x1, y1-30), replacement_text, font=self.font, fill="blue", align="center")

            image_path = os.path.join(self.out_data_dir, 'draw_images', f'{file.replace(".json", "")}')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            img.save(image_path)
            
            for question, item in data['questions'].items():
                options = item['answerTexts']
                answer_index = item['correctAnswer']
                value = options[answer_index]

                instruction = random.choice(instructions)
                instruction = instruction.replace('<key>', question).replace('<options>', str(options))
                file_name = os.path.abspath(image_path)
                metadata = {
                    "image": file_name,
                    "conversations": [
                        {'from': 'human', 'value': instruction},
                        {'from': 'gpt', 'value': f"{value}"},
                    ],
                }
                if split == 'train':
                    train.append(metadata)
                elif split == 'test':
                    test.append(metadata)

        for split, results in [('train', train), ('test', test)]:
            out_filepath = os.path.join(self.out_data_dir, f'{split}.json')        
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(results)}')
            with open(out_filepath, "w") as f:
                json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/ai2d', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/ai2d', type=str)
    parser.add_argument('--font_file', default='GoNotoCurrent.ttf', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()