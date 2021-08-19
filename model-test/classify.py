import os
import sys
import numpy as np

from common import openvino_infer

def paddle_infer(pdmodel_file, test_image):
    import paddle
    paddle.enable_static()

    # load model
    exe = paddle.static.Executor(paddle.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(
                        os.path.splitext(pdmodel_file)[0], # pdmodel prefix
                        exe)
    
    # print(feed_target_names, fetch_targets)

    # run
    output = exe.run(inference_program, feed={feed_target_names[0]: test_image}, fetch_list=fetch_targets)
    return output

def image_preprocess(img_path):
    import cv2

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = np.transpose(img, [2,0,1]) / 255
    img = np.expand_dims(img, 0)

    img_mean = np.array([0.485, 0.456,0.406]).reshape((3,1,1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    img -= img_mean
    img /= img_std

    return img.astype(np.float32)

def top_k(result, topk=5):
    indices = np.argsort(-result[0])

    # TopK
    for i in range(topk):
        print("classid:  ", indices[0][i], ", probability:  ", result[0][0][indices[0][i]], "\n")


def usage():
    print("\nUsage: python3 classify.py [test pdmodel] [test image] \n")

if __name__ == '__main__':
    import re

    abspath = os.path.abspath(__file__)
    dirs, _ = os.path.split(abspath)

    #
    usage()

    # model_path
    pdmodel_file = f'{dirs}/../paddle2ov/assets/pdmodels/openvino_support_clas/models/MobileNetV3_large_x1_0_infer/inference.pdmodel' # default
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]) and re.match(os.path.splitext(sys.argv[1])[-1],'.pdmodel$'):
        pdmodel_file = sys.argv[1]   
    print('info: test pdmode is {}'.format(pdmodel_file))

    # load Imge
    image_file = f'{dirs}/data/ILSVRC2012_val_00000003.JPEG' # default
    if len(sys.argv)>2 and os.path.isfile(sys.argv[2]):
        image_file = sys.argv[2]
    print('info: test image is {}'.format(image_file))

    test_image = image_preprocess(image_file) 

    # infer
    result_ie = openvino_infer(pdmodel_file, test_image)
    print('\ntopk of openvino:')
    top_k(result_ie)

    result_pd = paddle_infer(pdmodel_file, test_image)
    print('\ntopk of paddle:')
    top_k(result_pd)

   