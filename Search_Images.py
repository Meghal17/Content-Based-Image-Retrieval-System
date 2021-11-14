from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import random
import configparser
import os
import numpy as np
import cv2
import pickle
import shutil

def Search_Images(cluster_model, query_feature, N):
	'''
	query_feature : expected in np.array format with expanded dims along axis=0
	'''
	query_cluster = cluster_model.predict(query_feature)[0]
	cluster_dir = os.path.join('Clustered', str(query_cluster))
	img_names = random.sample(os.listdir(cluster_dir), N)
	
	ncols = 3
	nrows = math.ceil(N/ncols)
	fig, axs = plt.subplots(nrows, ncols, figsize = (17, 17))
	plt.subplots_adjust(hspace = 0.2, wspace=0.2)
	for i in range(nrows):
		for j in range(ncols):
			img_idx = (3*i) + j
			try:
				img_name = img_names[img_idx]
				img = plt.imread(os.path.join(cluster_dir, str(img_name)))
				p = fig.add_subplot(nrows,ncols, img_idx+1)
				p.imshow(img)
				p.axis('off')		#to turn off display of image dimensions
				axs[i,j].set_axis_off()	#to turn off the display of axis scale
			except:
				axs[i,j].set_axis_off()
				break
	plt.show()						#for matplotlib on colab via


cfg = configparser.ConfigParser()
c_path = "E:/CBIR/Config.txt"
# c_path =  '/content/gdrive/My Drive/CBIR/Approach2/ConfigDrive.txt'
cfg.read_file(open(c_path))

DATA_PATH = cfg.get('CBIR config','data_path')
MODEL_PATH = cfg.get('CBIR config','model_path')
QUERY_PATH = cfg.get('CBIR config','query_path')
INPUT_DIM = list(map(int, cfg.get('CBIR config','input_dim').split(',')))

print('[INFO] Loading Feature extractor model ...')
model = load_model(os.path.join(MODEL_PATH, 'ResNet50_extractor_model.h5'))
model.load_weights(os.path.join(MODEL_PATH, 'ResNet50_extractor_weights.h5'))
print('[INFO] Model loaded.')

n = int(input('Enter Image number for testing: '))
test_img = load_img(os.path.join(QUERY_PATH, str(n) + '.jpg'), target_size = (INPUT_DIM[0],INPUT_DIM[1]))
test_img = img_to_array(test_img)
test_img = preprocess_input(test_img)
test_img = np.expand_dims(test_img, axis = 0)

query_feat = model.predict(test_img).flatten()
query_feat = np.expand_dims(query_feat, axis=0)
query_num = int(input('Number of Images to be retreived: '))
clustering_model = pickle.load(open('K-Means model.sav', 'rb'))
sim_imgs = Search_Images(clustering_model, query_feat, query_num)