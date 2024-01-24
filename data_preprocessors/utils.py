import pandas as pd

def normalize_bbox(bbox, w=-1, h=-1):
    if w > 0 and h > 0:
        normalized_bbox = [
            int(1000 * bbox[0] / w),
            int(1000 * bbox[1] / h),
            int(1000 * bbox[2] / w),
            int(1000 * bbox[3] / h),
        ]
    else:
        normalized_bbox = [
            int(1000 * bbox[0]),
            int(1000 * bbox[1]),
            int(1000 * bbox[2]),
            int(1000 * bbox[3]),
        ]
        
    if len(bbox) == 4:
        return convert_wh(normalized_bbox)
    elif len(bbox) == 6:
        return normalized_bbox

def convert_wh(bbox):
    return [bbox[0], bbox[1], bbox[2], bbox[3], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])]

def sort_coordinate(bboxes):
    return sorted(bboxes , key=lambda k: [k[2][1], k[2][0]])    

def load_instructions(instruction_path):
    instructions = {}
    data = pd.read_excel(instruction_path)
    for d in data.values:
        dataset_name = d[0]
        insts = []
        for prompt in d[3:]:
            if pd.isna(prompt):
                break
            insts.append(prompt)
        instructions[dataset_name] = insts
    return instructions
