from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import time
import copy
    
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

from ppocr.utils import *
from ppocr.predict_det import TextDetector
from ppocr.predict_rec import TextRecognizer

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def pipeline(text_detector, text_recognizer, img, args):
        ori_im = img.copy()
        dt_boxes, elapse = text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))

        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= args.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res

def main(args):
    import re
    import cv2
    from PIL import Image

    # load Imge
    image_file = f'{__dir__}/data/image_1019.jpg' # default
    if args.image_dir and os.path.isfile(args.image_dir):
        image_file = args.image_dir
    print('info: test image is {}'.format(image_file))

    img = cv2.imread(image_file)

    '''
    ppocr det + rec pipeline
    '''
    text_detector = TextDetector(args)
    text_recognizer = TextRecognizer(args)
    
    st = time.time()
    dt_boxes, rec_res = pipeline(text_detector, text_recognizer, img, args)
    elapse = time.time() - st

    print("Predict time of {}: {}".format(image_file, elapse))
    for text, score in rec_res:
        print("{}, {:.3f}".format(text, score))

    if True:
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt(
            image,
            boxes,
            txts,
            scores,
            drop_score=args.drop_score,
            font_path=args.vis_font_path)
        draw_img_save = "./inference_results/"
        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)
        cv2.imwrite(
            os.path.join(draw_img_save, os.path.basename(image_file)),
            draw_img[:, :, ::-1])
        print("The visualized image saved in {}".format(
            os.path.join(draw_img_save, os.path.basename(image_file))))

if __name__ == "__main__":
    args = parse_args()
    main(args)  