import os
import json
import random
import argparse

train_val_datasets = ['klc', 'pwc', 'deepform', 'sroie', 'docile', 'wildreceipt', 'websrc', 'hwsquad',
                      'visualmrc', 'iconqa_fill_in_blank', 'iconqa_choose_txt', 'scienceqa',
                      'ai2d', 'docvqa', 'rvlcdip', 'textbookqa', 'wtq', 'tatdqa','scicap', 'llavar',
                      'screen2words', 'doclaynet', 'docbank', 'docvqa_iq', 'rvlcdip_io', 'ocrvqa']

def merge_datasets(input_data_dir='./processed_data', save_dir='./', max_samples=5000):
    questionId = 0
    for split in [('train'), ('dev', 'val')]:
        merge = []
        for dataset_name in train_val_datasets:
            for s in split:
                dataset_path = os.path.join(input_data_dir, dataset_name, f'{s}.json')
                if os.path.exists(dataset_path):
                    with open(dataset_path, 'r') as f:
                        data = json.load(f)
            if len(data) == 0:
                continue
            random.shuffle(data)[:max_samples]
            for d in data:
                d["dataset_name"] = dataset_name
                d["id"] = questionId
                merge.append(d)
        random.shuffle(merge)

        out_filepath = os.path.join(save_dir, f'{split[0]}.json')
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        print(f'{split}: {len(merge)}')
        with open(out_filepath, "w") as f:
            json.dump(merge, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='processed_data', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--max_samples', default=5000, type=int)
    args = parser.parse_args()

    merge_datasets(args.input_data_dir, args.save_dir, args.max_samples)