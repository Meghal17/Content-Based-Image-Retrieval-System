from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.cluster import KMeans
import configparser
import numpy as np
import cv2
import os
import pickle
import shutil

def build_model(input_dims):
	h = input_dims[0]
	w = input_dims[1]
	nc = input_dims[2]
	IN = Input(shape = (h,w,nc))
	model = ResNet50(include_top = False, weights='imagenet', input_tensor=IN)
	model = Model(inputs = model.input, outputs = model.get_layer('conv5_block3_2_relu').output)
	return model

def extract_features(d_path, batch_size, input_dims, model, n_images):
	Features = []
	for i in range(int(n_images/batch_size)+1):
		batch_images = []
		for j in range(i*batch_size, ((i+1)*batch_size)):
			if j < n_images:
				img_path = os.path.join(d_path, str(j) + '.jpg')
				img = load_img(img_path, target_size=(input_dims[0],input_dims[1]))
				img = img_to_array(img)
				img = preprocess_input(img)
				batch_images.append(img)
		batch_images = np.array(batch_images)
		features = model.predict(batch_images)
		for feature in features:
			Features.append(feature.flatten())
		if i%10 == 0:
			print('[INFO] Features from batch', i, 'extracted...')
	return Features

def Cluster_Images(Features, num_clusters, data_path):
	kmeans = KMeans(n_clusters=num_clusters, n_init = 50, max_iter = 500)
	kmeans.fit(Features)
	print('[INFO] Saving K-Means model ...')
	model_name = 'K-Means model.sav'
	pickle.dump(kmeans, open(model_name, 'wb'))
	print('[INFO] K-Means model saved.')
	clusters = [[] for _ in range(num_clusters)]
	for index, cluster_id in enumerate(kmeans.labels_):
		clusters[cluster_id].append(index)
	
	try:
		os.mkdir('Clustered')
		for i in range(num_clusters):
			os.mkdir('Clustered/' + str(i))
	except:
		pass

	for i in range(num_clusters):
		dest_path = os.path.join('Clustered',str(i))
		for img in clusters[i]:
			src_path = os.path.join(data_path, str(img) + '.jpg')
			shutil.copy(src_path, dest_path)

cfg = configparser.ConfigParser()
cfg.read_file(open('config.txt'))
DATA_PATH = cfg.get('CBIR config','data_path')
MODEL_PATH = cfg.get('CBIR config','model_path')
# FEATURES_PATH = cfg.get('CBIR config','features_path')		#Wont need this
INPUT_DIMS = list(map(int, cfg.get('CBIR config','input_dim').split(',')))
BATCH_SIZE = int(cfg.get('CBIR config','batch_size'))
N_IMAGES = len(os.listdir(DATA_PATH))
N_CLUSTERS = int(cfg.get('CBIR config', 'num_clusters'))

print('[INFO] Building ResNet model ...')
model = build_model(INPUT_DIMS)
model.save(os.path.join(MODEL_PATH, 'ResNet50_extractor_model.h5'))
model.save_weights(os.path.join(MODEL_PATH, 'ResNet50_extractor_weights.h5'))
print('[INFO] Model built and saved. ')

print('[INFO] Extracting features ... ')
features = np.array(extract_features(DATA_PATH, BATCH_SIZE, INPUT_DIMS, model, N_IMAGES))
print('[INFO] Features extracted. Now Clustering Dataset ...')
Cluster_Images(features, N_CLUSTERS, DATA_PATH)
print('[INFO] Data clustered.')