import numpy as np
import time
import csv
import sys
from collections import OrderedDict

from backend_processing import *
from frontend_processing import *
from graph_model import *
from incision_utils import *
import matplotlib.pyplot as plt


#keras
#import keras
# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.models import Model
# from keras import backend as K
# from keras.utils import plot_model
# from keras.models import load_model
#tf

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions



from tensorflow.keras.preprocessing import image

from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

def processing_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def inference(img_path,model):
    x = processing_img(img_path)
    start_time = time.time()
    preds = model.predict(x)
    result = decode_predictions(preds)
    # print(' Actual Predicted:', decode_predictions(preds, top=1)[0][0][1], '\n', 'Probabiity:',
    #      decode_predictions(preds, top=1)[0][0][2])
    return time.time() - start_time


def central_control(img_path,model,incision,file_name):
    
    x = processing_img(img_path)
    transfer_file_size,frontend_K_time,frontend_execution_time,frontend_save_time = transfer_processing(model,incision,x,file_name)
    #model_file = file_name + ".hdf5"
    npzfile = file_name + '_transfer_files.npz'
   
    donor_file_size,backend_load_time,backend_execution_time,backend_result_time = backend(model,npzfile)

    final_result_load_start_time = time.time()
    file = open("denor_files.txt", "r")
    result = file.read().split()
    #print('surgery:','\n', ' Actual Predicted:', result[0], '\n', 'Probabiity:',result[1])
    final_result_load_time  = time.time() - final_result_load_start_time
    
    return transfer_file_size,donor_file_size,frontend_K_time,frontend_execution_time,frontend_save_time,backend_load_time,backend_execution_time,backend_result_time,final_result_load_time


def multiple_surgery(img_path,file_name,frequency,feasible_wound,feasible_incision,model):
    key = np.arange(1,frequency+1).tolist()
    key.append('wound')


    with open('multi_latency_result_frontend.csv', mode='w') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=key)
        writer.writeheader()

        for i in range(len(feasible_wound)):
            wound = feasible_wound[i]
            incision = feasible_incision[i]

            multi_surgery_time_infor = {}
            multi_surgery_time_infor['wound'] = wound


            x = processing_img(img_path)
            frontend_execution_time = multi_transfer_processing(model, incision, x, file_name, frequency)
            for i in range(len(frontend_execution_time)):
                multi_surgery_time_infor[i+1] = frontend_execution_time[i]

            writer.writerow(multi_surgery_time_infor)

    with open('multi_latency_result_backend.csv', mode='w') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=key)
        writer.writeheader()

        for i in range(len(feasible_wound)):
            wound = feasible_wound[i]
            incision = feasible_incision[i]

            multi_surgery_time_infor = {}
            multi_surgery_time_infor['wound'] = wound

            x = processing_img(img_path)
            npzfile = file_name + '_transfer_files.npz'
            backend_execution_time = multi_backend(model, npzfile, frequency)

            for i in range(len(backend_execution_time)):
                multi_surgery_time_infor[i+1] = backend_execution_time[i]

            writer.writerow(multi_surgery_time_infor)


def re_multiple_surgery(img_path,file_name,frequency,feasible_wound,feasible_incision,model):
    key = np.arange(1,frequency+1).tolist()
    key.append('wound')


    with open('re_multi_latency_result_frontend.csv', mode='w') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=key)
        writer.writeheader()
        

        x = processing_img(img_path)

        for i in range(len(feasible_wound)):
            wound = feasible_wound[i]
            incision = feasible_incision[i]

            multi_surgery_time_infor = {}
            multi_surgery_time_infor['wound'] = wound

            cut, true_output_shape, layer_name_idx = generate_cut(model, incision)
            transplant = K.function([model.input], cut)
            execution_time = []

            for i in range(frequency):

                start_time = time.time()
                transplant_feature = transplant([x])
                multi_surgery_time_infor[i+1] = time.time() - start_time

            writer.writerow(multi_surgery_time_infor)

    with open('re_multi_latency_result_backend.csv', mode='w') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=key)
        writer.writeheader()
        
        x = processing_img(img_path)

        for i in range(len(feasible_wound)):
            wound = feasible_wound[i]
            incision = feasible_incision[i]

            multi_surgery_time_infor = {}
            multi_surgery_time_infor['wound'] = wound

            cut, true_output_shape, layer_name_idx = generate_cut(model, incision)
            transplant = K.function([model.input], cut)
            transplant_feature = transplant([x])
            get_layer_output = K.function(cut,[model.output])

            execution_time = []

            for i in range(frequency):

                start_time = time.time()
                layer_output = get_layer_output(transplant_feature)
                multi_surgery_time_infor[i+1] = time.time() - start_time

            writer.writerow(multi_surgery_time_infor)






def write_infor(feasible_wound,feasible_incision,img_path,model,file_name):
    execution_time_key = ['wound', 'inference_time',
                          'transfer_file_size', 'donor_file_size',
                          'frontend_K_time','frontend_execution_time','frontend_save_time', 
                          'backend_load_time','backend_execution_time','backend_result_time',
                          'frontend_time','backend_time','final_result_load_time','surgery_time']

    
    with open('latency_result_CPU_tensorflow.csv', mode='w') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=execution_time_key)
        writer.writeheader()
        for i in range(len(feasible_wound)):
            wound =  feasible_wound[i]
            incision = feasible_incision[i]
            surgery_time_infor = {}
            inference_time = inference(img_path, model)
            surgery_time_infor['wound'] = wound
            surgery_time_infor['inference_time'] = inference_time

            transfer_file_size,donor_file_size,frontend_K_time,frontend_execution_time,frontend_save_time,backend_load_time,backend_execution_time,backend_result_time,final_result_load_time = central_control(img_path,model,incision,file_name)


            surgery_time_infor['transfer_file_size'] = transfer_file_size
            surgery_time_infor['donor_file_size'] =  donor_file_size
            surgery_time_infor['frontend_K_time'] = frontend_K_time
            surgery_time_infor['frontend_execution_time'] = frontend_execution_time
            surgery_time_infor['frontend_save_time'] = frontend_save_time
            surgery_time_infor['frontend_time'] = frontend_execution_time + frontend_save_time

            surgery_time_infor['backend_load_time'] = backend_load_time
            surgery_time_infor['backend_execution_time'] = backend_execution_time
            surgery_time_infor['backend_result_time'] = backend_result_time
            surgery_time_infor['backend_time'] = backend_load_time + backend_execution_time + backend_result_time
            surgery_time_infor['final_result_load_time'] = final_result_load_time

            surgery_time_infor['surgery_time'] = surgery_time_infor['frontend_time'] + surgery_time_infor['backend_time'] + final_result_load_time

            writer.writerow(surgery_time_infor)





if __name__ == '__main__':
   
    file_name = "pre_trained_VGG19"
    print('load model..')
    model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    # model.save(file_name + ".hdf5")
    print('load inference image...')
    img_path = '/Users/yy/Documents/Study/USYD/COMP5707/final_code/elephant.jpg'


    # print('candidate incision prepared')
    # residule_no,inception_no,wound,incision = incision_generation(model)
    # print('there are {} residule modules, and {} inception modules'.format(residule_no,inception_no))

    #feasible_wound,feasible_incision = feasible_wound_generate(model,wound,incision,processing_img(img_path))

    #for Resnet50
    feasible_wound = [[1], [2], [3], [4], [5], [6], [7, 14], [7, 16], [8, 14], [8, 16], [9, 14], [9, 16], [10, 14], [10, 16], [11, 14], [11, 16], [12, 14], [12, 16], [13, 14], [13, 16], [15, 14], [15, 16], [17], [18], [27], [28], [37], [38], [39, 46], [39, 48], [40, 46], [40, 48], [41, 46], [41, 48], [42, 46], [42, 48], [43, 46], [43, 48], [44, 46], [44, 48], [45, 46], [45, 48], [47, 46], [47, 48], [49], [50], [59], [60], [69], [70], [79], [80], [81, 88], [81, 90], [82, 88], [82, 90], [83, 88], [83, 90], [84, 88], [84, 90], [85, 88], [85, 90], [86, 88], [86, 90], [87, 88], [87, 90], [89, 88], [89, 90], [91], [92], [101], [102], [111], [112], [121], [122], [131], [132], [141], [142], [143, 150], [143, 152], [144, 150], [144, 152], [145, 150], [145, 152], [146, 150], [146, 152], [147, 150], [147, 152], [148, 150], [148, 152], [149, 150], [149, 152], [151, 150], [151, 152], [153], [154], [163], [164], [173], [174], [175]]
    feasible_incision = wound2incision(feasible_wound)
    print('feasible incision prepared')
        

    print('start of basic surgury experiment')

    # write_infor(feasible_wound,feasible_incision,img_path,model,file_name)
    # print('end of basic surgury experiment')
    # print('start multiple sugury expeiment')
    re_multiple_surgery(img_path, file_name, 2, feasible_wound, feasible_incision, model)
    # print('end of multiple sugury expeiment')
    #print(feasible_wound,feasible_incision )
    
   