from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils import *
from common import openvino_infer

class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm

        pdmodel_file = f'{__dir__}/../../paddle2ov/assets/pdmodels/ocr_openvino_support/ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel' # default
        model_dir = args.rec_model_dir
        if model_dir and os.path.isfile(model_dir) and re.match(os.path.splitext(model_dir)[-1],'.pdmodel$'):
            pdmodel_file = model_dir
        print('info: det pdmode is {}'.format(pdmodel_file))        
        self.pdmodel_file = pdmodel_file

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def preprocess_op(self, data):       
        op1 = DetResizeForTest(limit_side_len = self.args.det_limit_side_len, limit_type = self.args.det_limit_type)
        op2 = NormalizeImage(
            std=[0.229, 0.224, 0.225],
            mean= [0.485, 0.456, 0.406],
            scale= '1./255.',
            order= 'hwc'
        )
        op3 = ToCHWImage()
        op4 = KeepKeys(keep_keys=['image', 'shape'])

        data = op4(op3(op2(op1(data))))

        return data
    
    def postprocess_op(self, preds, shape_list):
        if self.det_algorithm == "DB":
            op = DBPostProcess(
                thresh = self.args.det_db_thresh,
                box_thresh = self.args.det_db_box_thresh,
                max_candidates = 1000,
                unclip_ratio = self.args.det_db_unclip_ratio,
                use_dilation = self.args.use_dilation,
                score_mode = self.args.det_db_score_mode 
            )

            return op(preds, shape_list)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()

        # preprocess
        data = self.preprocess_op(data)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        # infer
        outputs = openvino_infer(self.pdmodel_file, img)
        # outputs = []
        # for output_tensor in self.output_tensors:
        #     output = output_tensor.copy_to_cpu()
        #     outputs.append(output)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs[0]
            preds['f_score'] = outputs[1]
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs[0]
            preds['f_score'] = outputs[1]
            preds['f_tco'] = outputs[2]
            preds['f_tvo'] = outputs[3]
        elif self.det_algorithm == 'DB':
            preds['maps'] = outputs[0]
        else:
            raise NotImplementedError

        # postprocess
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        if self.det_algorithm == "SAST" and self.det_sast_polygon:
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et = time.time()
        return dt_boxes, et - st        

def main(args):
    import re
    import cv2

    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)    

    # load Imge
    image_file = f'{__dir__}/../data/image_1019.jpg' # default
    if args.image_dir and os.path.isfile(args.image_dir):
        image_file = args.image_dir
    print('info: test image is {}'.format(image_file))

    img = cv2.imread(image_file)

    text_detector = TextDetector(args)

    st = time.time()
    dt_boxes, _ = text_detector(img)
    elapse = time.time() - st

    print("Predict time of {}: {}".format(image_file, elapse))
    src_im = draw_text_det_res(dt_boxes, image_file)
    img_name_pure = os.path.split(image_file)[-1]
    img_path = os.path.join(draw_img_save,
                            "det_res_{}".format(img_name_pure))
    cv2.imwrite(img_path, src_im)
    print("The visualized image saved in {}".format(img_path))    

if __name__ == "__main__":
    args = parse_args()
    main(args)  