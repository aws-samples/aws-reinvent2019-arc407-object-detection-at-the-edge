#
# Copyright 2010-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import mxnet as mx
import numpy as np
import cv2
import urllib2
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

class SSDResnetModel(object):

    def __init__(self, network_prefix, params_url=None, context=mx.cpu(),
                 input_shapes=[('data', (1, 3, 512, 512))]):
        print("Loading the model")
        if params_url is not None:
            fetched_file = urllib2.urlopen(params_url)
            with open(network_prefix + "-0000.params", 'wb') as output:
                output.write(fetched_file.read())
        print("loading checkpoint")
        # Load the network parameters from default epoch 0
        sym, arg_params, aux_params = mx.model.load_checkpoint(network_prefix, 0)
        print("loading mod..")
        # Load the network into an MXNet module and bind the corresponding parameters
        self.mod = mx.mod.Module(symbol=sym, label_names=None, context=context)
        self.mod.bind(for_training=False, data_shapes=input_shapes)
        self.mod.set_params(arg_params, aux_params)
        self.camera = None
        print("done initializing..")

    def predict_from_file(self, filepath, reshape=(512,512)):
        topN = []
        print("reading image:" + filepath)
        cvimage = cv2.imread(filepath)
        # Switch RGB to BGR format
        img = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
        if img is None:
            return topN

        print("resizing the image now")
        # Resize image to fit network input
        img = cv2.resize(img, reshape)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]

        print("running the image through model")
        # Run forward on the image
        self.mod.forward(Batch([mx.nd.array(img)]))
        prob = self.mod.get_outputs()[0].asnumpy()
        print("probabilities: " + str(prob))
        detections = prob
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)

        dets = result[0]
        width = reshape[0]    # original input image width
        height = reshape[1]  # original input image height
        response = []
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > 0.2:
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    class_name = str(cls_id)
                    response.append((class_name, xmin, ymin, xmax, ymax))
        return response