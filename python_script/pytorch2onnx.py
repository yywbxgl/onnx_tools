from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.onnx as torch_onnx
from torch.autograd import Variable
from onnxruntime.backend import prepare
import onnx
import numpy as np

from torchvision.models.googlenet import *
from torchvision.models.mobilenet import *
from torchvision.models.shufflenetv2 import *

import sys
sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="path_to_imagenet",
                    help='path_to_imagenet')
############################################################################
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnetamobile',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument("--model", default='', help="input model direction")
############################################################################
parser.add_argument('--pretrained', default='imagenet',
                    help='use pre-trained model')
parser.add_argument('--result', '-r', metavar='result_print', default='N',
                    help='print the predict result or not')
############################################################################
parser.add_argument('--h', metavar='size_input_height', default= 224,
                    help='the size of input data')
parser.add_argument('--w', metavar='size_input_width', default= 224,
                    help='the size of input data')
############################################################################
parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # step 1.1, create pretrained model to transform to onnx model
    print("=> creating model '{}'".format(args.arch))
    if args.model == '':
        if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
            print("=> using pre-trained parameters '{}'".format(args.pretrained))
            Model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                         pretrained=args.pretrained)
        else:
            Model = pretrainedmodels.args.arch

    if args.model == 'mobilenet_v2' or args.model == 'mobilenetv2':
        Model = mobilenet_v2(pretrained = True)
    if args.model == 'googlenet':
        Model = googlenet(pretrained = True)
    if args.model == 'shufflenet_v2' or args.model == 'shufflenetv2':
        Model = shufflenet_v2_x0_5(pretrained = True)

    # Use this an input trace to serialize the model
    input_shape = (3, int(args.h), int(args.w))
    if args.model == '':
        modelname = args.arch
    else:
        modelname = args.model
    model_onnx_path = "models_converted/%s-pytorch.onnx"%modelname
    model = Model
    if args.model == '':
        model.train(False)

    # step 1.2 Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))

    torch.save(model, 'models_converted/%s.path'%modelname)

    torch_onnx.export(model,
                      dummy_input,
                      model_onnx_path,
                      verbose=False)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("=> Export of %s-pytorch.onnx complete!"%modelname)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    if args.model !='':
        print('The checking processing only support the <arch> mode.')
    else:
        # step 2, create onnx_model using onnxruntime as backend. check if right and export graph.
        image_path = 'data/2127.jpg'

        # output_pytorch, img_np = modelhandle.process(image)

        load_img = pretrainedmodels.utils.LoadImage()

        # transformations depending on the model
        # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
        tf_img = pretrainedmodels.utils.TransformImage(model)

        # path_img = 'data/cat.jpg'

        input_img = load_img(image_path)
        input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
        input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
        input = torch.autograd.Variable(input_tensor,
                                        requires_grad=False)

        # step 2.1 output the result of the test with pytorch
        output_pytorch = model(input)  # 1x1000
        # print('output_pytorch = {}'.format(output_pytorch))

        # step 2.2 output the result of test with onnx
        img_np = input.cpu().detach().numpy()
        print('=> The size of input is ',img_np.shape)
        onnx_model = onnx.load(model_onnx_path)
        rep = prepare(onnx_model, strict=False)
        output_onnx = rep.run(img_np)
        output_onnx = np.mean(output_onnx, axis=1)
        # print(output_onnx_tf)
        # print('output_onnx_tf = {}'.format(output_onnx))

        #step 2.3 output the different of result between two models
        diff = output_onnx - output_pytorch.detach().numpy()
        diff = np.mean(diff, axis=1)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print("=> the precentage of different is %f "%diff[0])
        if diff <= 0.01:
            print("=> Model Exporting is Successful")
            print("=> Please check the model in direction 'models_converted/%s-pytorch.onnx'"%modelname)
        else:
            print('=> Model Exporitng may have problem, please check again')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        n = args.result
        if n == 'Y' or n == 'y':
            print('output_pytorch = {}'.format(output_pytorch))
            print('output_onnx_tf = {}'.format(output_onnx))


if __name__ == '__main__':
    main()

