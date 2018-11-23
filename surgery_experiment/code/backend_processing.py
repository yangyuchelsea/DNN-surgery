

import numpy as np
import os
import time
#keras
# from keras import backend as K
# from keras import Model
# from keras.models import load_model
# from keras.applications.resnet50 import decode_predictions
#tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import decode_predictions



def backend(model,npzfile):
    #model = load_model(model_file)
    backend_start_time = time.time()
    npzfile = np.load(npzfile)
  
    backend_instruct = []
    for i in range(len(npzfile[npzfile.files[0]])):
        backend_instruct.append((npzfile[npzfile.files[0]][i][0],int(npzfile[npzfile.files[0]][i][1])))
    backend_incision = []
    for i in backend_instruct:
        if i[1] == 0:
            backend_incision.append(model.layers[i[0]].input)
        elif i[1] == 1:
            backend_incision.append(model.layers[i[0]].output)
        else:
            print('incorrect wound, check cut layers')
    backend_transplant_feature = []
    for i in range(1,len(npzfile.files)):
        layer_feature = []
        layer_feature.append(npzfile[npzfile.files[i]].tolist())
        backend_transplant_feature.append(np.asarray(layer_feature))

    backend_load_time =  time.time() - backend_start_time

    backend_execution_start_time = time.time()

    get_layer_output = K.function(backend_incision,[model.output])
    layer_output = get_layer_output(backend_transplant_feature)

    backend_execution_time = time.time() - backend_execution_start_time

    backend_result_start_time = time.time()
    result = decode_predictions(layer_output[0], top=1)
    file = open('denor_files.txt', 'w')
    file.write(result[0][0][1])
    file.write(' ')
    file.write(str(result[0][0][2]))
    file.close()
    statinfo = os.stat('denor_files.txt')

    backend_result_time = time.time() - backend_result_start_time
    
    return statinfo.st_size,backend_load_time,backend_execution_time,backend_result_time


def multi_backend(model, npzfile,frequency):
    # model = load_model(model_file)
    npzfile = np.load(npzfile)

    backend_instruct = []
    for i in range(len(npzfile[npzfile.files[0]])):
        backend_instruct.append((npzfile[npzfile.files[0]][i][0], int(npzfile[npzfile.files[0]][i][1])))
    backend_incision = []
    for i in backend_instruct:
        if i[1] == 0:
            backend_incision.append(model.layers[i[0]].input)
        elif i[1] == 1:
            backend_incision.append(model.layers[i[0]].output)
        else:
            print('incorrect wound, check cut layers')
    backend_transplant_feature = []
    for i in range(1, len(npzfile.files)):
        layer_feature = []
        layer_feature.append(npzfile[npzfile.files[i]].tolist())
        backend_transplant_feature.append(np.asarray(layer_feature))

    get_layer_output = K.function(backend_incision, [model.output])

    execution_time = []
    for i in range(frequency):
        start_time = time.time()
        layer_output = get_layer_output(backend_transplant_feature)
        execution_time.append(time.time() - start_time)

    result = decode_predictions(layer_output[0], top=1)
    file = open('denor_files.txt', 'w')
    file.write(result[0][0][1])
    file.write(' ')
    file.write(str(result[0][0][2]))
    file.close()

    return execution_time



