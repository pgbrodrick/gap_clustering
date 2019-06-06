
import numpy as np
import sys
import subprocess
import os
import signal
from scipy import stats
import subprocess
import matplotlib.pyplot as plt
import math
import scipy.sparse.linalg

from scipy import linalg

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pandas as pd

import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator


MUNGED_DIR = 'kmunged'
OUTPUT_DIR = 'kmeans_output'
if (os.path.isdir(MUNGED_DIR) is False):
    os.mkdir(MUNGED_DIR)
if (os.path.isdir(OUTPUT_DIR) is False):
    os.mkdir(OUTPUT_DIR)


# Assumed to be a numpy file
input_numpy_file = sys.argv[1]

# Number of clusters
n_c = int(sys.argv[1])

csvf = os.path.join(MUNGED_DIR,'{}_b_{}.csv'.format(os.path.basename(input_numpy_file).split('.')[0],b)
output_file = os.path.join(OUTPUT_DIR,'kmeans_results_nc_{}_b_{}.npz'.format(n_c,b)

if (os.path.isfile(csvf) == False):
    full_dat = np.load(input_numpy_file)
    full_dat = np.squeeze(full_dat[:,good_bands_b,res])

    # rescale data (not in original paper, but necessary to make histograms match
    scaler= preprocessing.StandardScaler()
    full_dat = scaler.fit_transform(full_dat)
      

    if (b != 0):
      np.random.seed(b)
      perm = np.random.permutation(len(full_dat))
      full_dat = full_dat[perm,:]
      print('transform...')
      #u,s,vt = scipy.sparse.linalg.svds(full_dat,full_dat.shape[1])
      u,s,vt = scipy.linalg.svd(full_dat,full_matrices=False)
      Xb = np.matrix(full_dat) * np.matrix(np.transpose(vt))
      full_dat_mins = np.squeeze(np.array(np.min(Xb,axis=0)))
      full_dat_maxs = np.squeeze(np.array(np.max(Xb,axis=0)))
      
      print(full_dat_mins.shape)
      print(full_dat_maxs.shape)
      
      del Xb
      
      for n in range(0,full_dat.shape[1]):
        full_dat[:,n] = np.random.uniform(full_dat_mins[n],full_dat_maxs[n],full_dat[:,n].shape)

      # back transform to get final set
      full_dat = np.matrix(full_dat)*np.matrix(vt)

      print('transformed')
      pd.DataFrame(full_dat).to_csv(csvf,index=False)
      del full_dat
    else: # b == 0
      pd.DataFrame(full_dat).to_csv(csvf,index=False)
      del full_dat

h2o.init()
train_frame = h2o.import_file(csvf)
model = H2OKMeansEstimator(k=n_c, init="PlusPlus", seed=10,max_iterations=10000)
model.train(x=train_frame.col_names,training_frame=train_frame)
model.summary()
#h2o.save_model(model,output_model_name)
Wk_tot = model.tot_withinss() / (2*train_frame.shape[0])


np.savez(output_file,
         n_c=n_c,
         b=b,
         Wk_tot=Wk_tot)

























