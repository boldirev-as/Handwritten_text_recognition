import json
import os
import sys
import time

import cv2
import numpy as np

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sentencepiece
import pickle

from contextlib import contextmanager

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGM_MODEL_PATH = "segm_model_final.pth"
OCR_MODEL_PATH = "model-last.pt"


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


class SEGMpredictor:
    def __init__(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = 600
        cfg.INPUT.MAX_SIZE_TEST = 800
        cfg.INPUT.FORMAT = "BGR"
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        outputs = self.predictor(img)
        prediction = outputs["instances"].pred_masks.cpu().numpy()
        contours = []
        for pred in prediction:
            contour_list = get_contours_from_mask(pred)
            contours.append(get_larger_contour(contour_list))
        return contours


class CustomTrOCRProcessor:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def __call__(self, *args, **kwargs):
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor


class OcrPredictor:
    def __init__(self, model_path):
        self.weights_dict = torch.load(model_path)
        self.model = torch.load('./model1.pth', map_location=DEVICE)
        self.model.load_state_dict(self.weights_dict['torch'])
        self.model.eval()
        self.model.cuda()
        self.model = self.model.half()
        self.processor = pickle.load(open('./processor.pkl', 'rb'))

    def __call__(self, image):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values.half().cuda())  # [0].cuda(non_blocking=True))
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text


def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y: y + h, x: x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


def main():
    t = time.time()
    pred_data = {}
    segm_predictor = SEGMpredictor(model_path=SEGM_MODEL_PATH)
    ocr_predictor = OcrPredictor(model_path=OCR_MODEL_PATH)

    for img_name in os.listdir(TEST_IMAGES_PATH):
        image = cv2.imread(os.path.join(TEST_IMAGES_PATH, img_name))

        output = {"predictions": []}
        contours = segm_predictor(image)
        for contour in contours:
            if contour is not None:
                crop = crop_img_by_polygon(image, contour)
                pred_text = ocr_predictor(crop)
                output["predictions"].append(
                    {
                        "polygon": [[int(i[0][0]), int(i[0][1])] for i in contour],
                        "text": pred_text,
                    }
                )

        pred_data[img_name] = output

        if (time.time() - t) / 60 >= 20:
            break

    with open(SAVE_PATH, "w") as f:
        json.dump(pred_data, f)


if __name__ == "__main__":
    main()
