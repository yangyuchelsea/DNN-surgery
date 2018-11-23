
from collections import OrderedDict
import numpy as np
import os
import time
from incision_utils import *

#keras
# from keras.preprocessing import image
# from keras.models import Model
# from keras import backend as K
# from keras.utils import plot_model as plot_model_without_idx

#tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model as plot_model_without_idx



def save_transfer_files(transplant_feature,true_output_shape,transfer_files_name,layer_name_idx,incision):
    interdiate_output_shape = []
    #transplant_feature = transplant_feature[0]
    for i in range(len(transplant_feature)):
        interdiate_output_shape.append(transplant_feature[i].shape[1:])

    if true_output_shape == interdiate_output_shape:
        wound_params = OrderedDict([('instruct',np.asarray(incision))])
        transplant_feature_params = OrderedDict([(layer_name_idx[incision[i][0]], transplant_feature[i][0]) for i in range(len(incision))])

        np.savez(transfer_files_name,**wound_params,**transplant_feature_params)
        statinfo = os.stat(transfer_files_name)
    else:
        print('incorrect output size',true_output_shape,interdiate_output_shape)

    return statinfo.st_size


def transfer_processing(model,incision,x,file_name):

    #plot_model(idx_flag = True, model = model, to_file = file_name+'idx.png', show_shapes=True, show_layer_names=True)
    #0: input 1: output
    frontend_start_time = time.time()
    cut, true_output_shape, layer_name_idx = generate_cut(model,incision)
    transplant = K.function([model.input], cut)

    frontend_K_time = time.time() - frontend_start_time

    frontend_execution_start_time = time.time()
    transplant_feature = transplant([x])
    
    print(isinstance(transplant_feature, list))
    frontend_execution_time= time.time() - frontend_execution_start_time
    frontend_save_start_time = time.time()
    transfer_files_name = file_name+'_transfer_files.npz'
    transfer_file_size = save_transfer_files(transplant_feature,true_output_shape,transfer_files_name,layer_name_idx,incision)
    frontend_save_time = time.time() - frontend_save_start_time

    return transfer_file_size,frontend_K_time,frontend_execution_time,frontend_save_time 

def multi_transfer_processing(model,incision,x,file_name,frequency):

    #0: input 1: output
    cut, true_output_shape, layer_name_idx = generate_cut(model, incision)
    transplant = K.function([model.input], cut)
    execution_time = []
    for i in range(frequency):

        start_time = time.time()
        transplant_feature = transplant([x])
        execution_time.append(time.time() - start_time)


    transfer_files_name = file_name+'_transfer_files.npz'
    transfer_file_size = save_transfer_files(transplant_feature,true_output_shape,transfer_files_name,layer_name_idx,incision)

    return execution_time
