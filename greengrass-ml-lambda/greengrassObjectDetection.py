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

# greengrassObjectDetection.py
# Demonstrates inference at edge using MXNET, SSD Resnet50 model
# and Greengrass core sdk. This function will continuously retrieve the
# predictions from the ML framework and send them to the topic 'counts'.
# The function will sleep for three seconds, then repeat.  Since the function is
# long-lived it will run forever when deployed to a Greengrass core.  The handler
# will NOT be invoked in our example since the we are executing an infinite loop.
#
# Prerequisites:
#
# MXNET: Please refer AWS Greengrass documentation for a proper installation
# of MXNET on Raspberry-Pi and how to bundle the output with your lambda.
# Installation script will take of all the other dependencies needed to run
# this inference code on your GGC.
#
# MODEL: "Resnet-50"
# This lambda expects to have
#       "model_algo_1-0000.params",
#       "model_algo_1-symbol.json"
#       and "hyperparams.json" files
# on the model_path folder of your project.
# DEVICE ACCESS: Please use AWS Greengrass console, CLI or API to add the below
# local device resources to your Greengrass group and then attach them to this
# lambda using the below settings also:
#
#      "Path": "/dev/vcsm", "Access": "r", "AutoAddGroupOwner": true },
#      "Path": "/dev/vchiq", "Access": "r", "AutoAddGroupOwner": true }

import sys
import time
import greengrasssdk
import platform
from threading import Timer
import load_model
from collections import Counter

client = greengrasssdk.client('iot-data')
model_path = '/models/object-detection/model/'
global_model = load_model.SSDResnetModel(model_path + 'model_algo_1')

# When deployed to a Greengrass core, this code will be executed immediately
# as a long-lived lambda function.  The code will enter the infinite while loop
# below.
def greengrass_object_detection_run():
    object_categories = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']

    if global_model is not None:
        try:
            print("trying to predict now...")
            predictions = global_model.predict_from_file("/images/sample.jpeg")
            # predictions =
            print predictions
            # publish predictions
            client.publish(topic='counts', payload='New Prediction: {}'.format(str(predictions)))
            counts = Counter([object_categories[int(k[0])] for k in predictions])
            print counts
            client.publish(topic='counts', payload='Counts: {}'.format(str(counts)))

        except:
            e = sys.exc_info()[0]
            print("Exception occured during prediction, oh yeah: %s" % e)

    # Asynchronously schedule this function to be run again in 3 seconds
    Timer(3, greengrass_object_detection_run).start()


# Execute the function above
greengrass_object_detection_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return