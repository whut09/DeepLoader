# -*- coding:utf-8 -*-
import os
import numpy as np


def cosine_similarity(v1, v2):
    cosV12 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosV12


def cosine_distance(v1, v2):
    return 1 - cosine_similarity(v1, v2)


def L2_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


def makedirs(path):
    dir, fname = os.path.split(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            pass


def check_model_outputs(extractor, session, input_shape, var_shape):
    check_rounds = 3
    input_shape = input_shape[0]
    # get the name of the first input of the model
    input_name = session.get_inputs()[0]
    print('Test for round:%d' % check_rounds)
    for i in range(check_rounds):
        if var_shape and var_shape[0][2] == '?':
            input = np.random.randn(2, input_shape[1], input_shape[2] * 4,
                                    input_shape[3] * 4).astype(
                np.float32) * 2 - 1
        elif var_shape and var_shape[0][0] == '?':
            input = np.random.randn(2, input_shape[1], input_shape[2],
                                    input_shape[3]).astype(
                np.float32) * 2 - 1
        else:
            input = np.random.randn(*input_shape).astype(
                np.float32) * 2 - 1
        # run x
        out1 = extractor.run([], {input_name: input})
        if i == 0:
            print('Outputs  ori:')
            for idx, out in enumerate(out1):
                print('  {} {}'.format(idx, out.shape))

        # run onnx
        out2 = session.run([], {input_name: input})
        if i == 0:
            print('Outputs onnx:')
            for idx, out in enumerate(out2):
                print('  {} {}'.format(idx, out.shape))

        assert len(out1) == len(out2)
        if i == 0:
            for idx in range(len(out1)):
                f1 = out1[idx]
                f1 = f1.flatten()
                f2 = out2[idx]
                f2 = f2.flatten()
                # compare
                dist_l2 = L2_distance(f1, f2)
                dist_cosine = cosine_distance(f1, f2)
                dim = f1.shape[0]
                data_norm = min(np.linalg.norm(f1), np.linalg.norm(f2))
                data_norm = max(data_norm, 1)
                print('  Output:%d norm:%f dim:%d' % (idx, data_norm, dim))
                print('  Cosine:%f' % dist_cosine)
                print('      L2:%f' % dist_l2)
                assert abs(dist_l2) < 1e-4 * data_norm
                assert abs(dist_cosine) < 1e-4

    print('Pass')
    return True
