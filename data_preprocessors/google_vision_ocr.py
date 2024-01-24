import base64
import json
from requests import Request, Session
from io import BytesIO
from utils import normalize_bbox

class Google_OCR:
    def __init__(self, api_key):
        self.api_key = api_key

    def pil_image_to_base64(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        str_encode_file = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return str_encode_file

    def recognize_image(self, pil_image):
        str_encode_file = self.pil_image_to_base64(pil_image)
        str_url = "https://vision.googleapis.com/v1/images:annotate?key="
        str_headers = {'Content-Type': 'application/json'}
        str_json_data = {
            'requests': [
                {
                    'image': {
                        'content': str_encode_file
                    },
                    'features': [
                        {
                            'type': "TEXT_DETECTION",
                        }
                    ]
                }
            ]
        }

        obj_session = Session()
        obj_request = Request("POST",
                              str_url + self.api_key,
                              data=json.dumps(str_json_data),
                              headers=str_headers
                              )
        obj_prepped = obj_session.prepare_request(obj_request)
        obj_response = obj_session.send(obj_prepped,
                                        verify=True,
                                        timeout=60
                                        )

        if obj_response.status_code == 200:
            return obj_response.json()

        else:
            return "error"

    def extract_info(self, items, img_w, img_h):
        words = []
        bboxes = []
        for page_ocrs in items['responses'][0]['fullTextAnnotation']['pages']:
            for block_ocrs in page_ocrs['blocks']:
                for para_ocrs in block_ocrs['paragraphs']:
                    for word_ocrs in para_ocrs['words']:
                        char_bboxes = []
                        word = ''
                        for sym_ocrs in word_ocrs['symbols']:
                            try:
                                bbox = sym_ocrs['boundingBox']
                                xmin = max(0, bbox['vertices'][0]['x'])
                                ymin = max(0, bbox['vertices'][0]['y'])
                                xmax = max(0, bbox['vertices'][2]['x'])
                                ymax = max(0, bbox['vertices'][2]['y'])
                                bbox = [xmin, ymin, xmax, ymax]
                            except:
                                continue
                            word += sym_ocrs['text']
                            char_bboxes.append(bbox)
                        if len(char_bboxes) > 0:
                            x1 = [w_p[0] for w_p in char_bboxes]
                            y1 = [w_p[1] for w_p in char_bboxes]
                            x2 = [w_p[2] for w_p in char_bboxes]
                            y2 = [w_p[3] for w_p in char_bboxes]
                            word_bbox = [min(x1), min(y1), max(x2), max(y2)]
                            if word_bbox[0] >= word_bbox[2] or word_bbox[1] >= word_bbox[3]:
                                continue
                            word_bbox = normalize_bbox(word_bbox, img_w, img_h)
                            words.append(word)
                            bboxes.append(word_bbox)
        return words, bboxes
