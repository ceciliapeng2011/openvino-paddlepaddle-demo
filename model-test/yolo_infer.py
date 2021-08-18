import os, sys
import numpy as np
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork

abspath = os.path.abspath(__file__)
dirs, _ = os.path.split(abspath)


# model_path
pdmodel_path = f'{dirs}/../paddle2ov/assets/pdmodels/yolo/ppyolo_r50vd_dcn_1x_coco/model.pdmodel'

device = 'CPU'

ie = IECore()
#ie.set_config({"DYN_BATCH_ENABLED": "YES"}, device)
net = ie.read_network(pdmodel_path)

net.reshape({'image': [1, 3, 608, 608], 'im_shape': [
            1, 2], 'scale_factor': [1, 2]})

exec_net = ie.load_network(net, device)
assert isinstance(exec_net, ExecutableNetwork)

print(net.input_info)
print(net.outputs)


test_image = np.random.rand(1, 3, 608, 608).astype('float32')
test_im_shape = np.array([[608, 608]]).astype('float32')
test_scale_factor = np.array([[0.95, 1.423887]]).astype('float32')

inputs_dict = {'image': test_image, "im_shape": test_im_shape,
               "scale_factor": test_scale_factor}
#request = exec_net.requests[0]
# request.set_batch(1)
output = exec_net.infer(inputs_dict)
