def openvino_infer(pdmodel_file, test_image, result_type='list'):
    from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork

    ie = IECore()
    net = ie.read_network(pdmodel_file)

    print(net.input_info.keys())
    print(net.outputs.keys())
    print(test_image.shape)
    
    # pdmodel might be dynamic shape
    input_key = list(net.input_info.items())[0][0] # 'x'
    net.reshape({input_key: test_image.shape})

    exec_net = ie.load_network(net, 'CPU') # device
    assert isinstance(exec_net, ExecutableNetwork)

    output = exec_net.infer({input_key: test_image})

    if result_type == 'list': # classification
        print(list(output.items())[0][1])
        return list(output.values())
    else: # ppocr-rec
        print(list(output.items())[0][1].shape)
        return list(output.items())[0][1]