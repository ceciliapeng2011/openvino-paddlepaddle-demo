import argparse
import os
import sys
import numpy as np

# openvino python api
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

def main():
    args = parse_args()

    # model_path
    test_model = os.path.abspath(args.model_file)

    ie = IECore()
    net = ie.read_network(args.model_file)

    net.reshape({'image': [1, 3, 608, 608], 'im_shape': [1, 2], 'scale_factor': [1, 2]})

    exec_net = ie.load_network(net, 'CPU')
    assert isinstance(exec_net, ExecutableNetwork)

    print(net.input_info)
    print(net.outputs)

    test_image = np.random.rand(args.batch_size, 3, 608, 608).astype('float32')
    test_im_shape = np.array([[608, 608]]).astype('float32')
    test_scale_factor = np.array([[0.95, 1.423887]]).astype('float32')

    inputs_dict = {'image': test_image, "im_shape": test_im_shape, "scale_factor": test_scale_factor}
    output = exec_net.infer(inputs_dict)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, default=os.path.join(__dir__, '../../PaddleOMZAnalyzer/exporter/paddledet/ppyolo_r50vd_dcn_1x_coco/model.pdmodel'), help="model filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args() 

if __name__ == "__main__":
    main()    
