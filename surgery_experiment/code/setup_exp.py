import numpy as np
import time
import csv

from graph_model import *
from incision_utils import *
import matplotlib.pyplot as plt

#keras
#import keras
#import keras.applications as app
# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.models import Model
# from keras import backend as K
# from keras.utils import plot_model
# from keras.models import load_model
#tf

import tensorflow as tf
import tensorflow.keras.applications as app
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
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


def model_exp(frequency):
	for i in range(frequency):
		load_time = []
		start = time.time()
		#set the model
		model = app.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
		load_time.append(time.time()-start)
	print('mean of load time:',np.mean(load_time))
	plt.plot(load_time,'bo--')
	plt.xlabel('frequency')
	plt.ylabel('load time')
	plt.title('Time of load model')
	plt.show()

def one_image_inference_exp(img,model,frequency,file_name):
	time_list = []
	for i in range(frequency):
		time_list.append(inference(img,model))
	print('first inference time',time_list[0],'mean of inference time from second time',np.mean(time_list[1:]))
	plt.plot(time_list)
	plt.xlabel('frequency')
	plt.ylabel('predict time')
	plt.title('Time of predict elephant image')
	plt.show()

	# with open(file_name, 'w') as f:
	# 	for item in time_list:
	# 		f.write("%s\n" % item)

def multi_image_inference_exp(img_path,inference_image,model,frequency):
	predict_time = []
	for index in range(len(inference_image)):
		img = image.load_img(img_path + inference_image[index] + '.jpg', target_size=(224,224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		start = time.time()
		preds = model.predict(x)# make predictions then decode and print them
		result = decode_predictions(preds, top=3)[0]
		predict_time.append(time.time()-start)

	plt.plot(inference_image,predict_time,'bo--')
	plt.xlabel('images')
	plt.ylabel('inference time')
	plt.title('predict time of different image')
	plt.xlabel('images')
	plt.ylabel('inference time')
	plt.title('predict time of different image')
	plt.show()


def surgery_basic_exp(img_path,model,wound):
	start = time.time()
	transplant = K.function([model.input],[model.layers[wound].output])
	print('frontend function load time',time.time()-start)
	start = time.time()
	transplant_backend = K.function([model.layers[wound].output],[model.output])
	print('backend function load time',time.time()-start)

	x = processing_img(img_path)
	frontend_inference_time = []
	backend_inference_time = []
	inference_with_sugery = []
	inference_time =[]

	for i in range(10):
		inference_time.append(inference(img_path,model))
		start = time.time()
		predict = transplant([x])
		frontend_inference_time.append(time.time()-start)
		start_2 = time.time()
		predict_1 = transplant_backend(predict)
		backend_inference_time.append(time.time()-start_2)

		result = decode_predictions(predict_1[0], top=3)[0]
		inference_with_sugery.append(time.time()-start)

	plt.plot(inference_with_sugery)
	plt.plot(inference_time)
	plt.plot(frontend_inference_time)
	plt.plot(backend_inference_time)
	plt.xlabel('frequency')
	plt.ylabel('predict time')
	plt.title('sugery predict time')
	plt.legend(['sugery predict time ','predict time','frontend predict time','backend predict time'], loc='upper right')
	plt.show()

if __name__ == '__main__':

	print('load model..')
	model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
					 classes=1000)
	# model.save(file_name + ".hdf5")

	file_name = "pre_trained_ResNet50"
	print('load image')
	img_path = '/Users/yy/Documents/Study/USYD/COMP5707/code/inference_image/'
	inference_image = ['elephant','cat','horse','toilet_tissue','bike','cup','woman']
	# #exp1.test load model time
	# #input: frequency(time of load model)
	# #outpu:mean of load time, figure of load time
	# model_exp(2)

    # #exp2.test inference time of single image
	# #input: image path,file name and frequency(time of predict)
	# #output: first inference time,mean of inference time except the first time,a figure of inference time and a txt of inference time
	# one_image_inference_exp(img_path + inference_image[0] + '.jpg',model,10,'inference_time.txt')

	# #exp3.test inference time of multiple images
	# #input: image path,images and frequency(time of predict)
    # #output: a figure of inference time
	# multi_image_inference_exp(img_path, inference_image, model, 2)


    # #exp4. Given a single wound, test execution of each process
	# #input:image,wound
	# #output:frontend function load time,backend function load time,
	# #       and a figure of inference time(sugery,normal,frontend and backend)
	# surgery_basic_exp(img_path + inference_image[0] + '.jpg', model,91)




