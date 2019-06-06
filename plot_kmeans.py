import numpy as np
import sys
import subprocess
import os
import signal
from scipy import stats
import subprocess

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import factorial


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



INPUT_DIR = 'kmeans_output'
FIG_DIR = 'figs'
nc_range = range(2,501)


res = 0

gap = []
s_k = []
wk = []
wkb = []
wkb_sd = []
infile_base = os.path.join(OUTPUT_DIR,'kmeans_results_nc_'

n_cs = []
for n_c in nc_range:
 const = 1 # Handled during Wk calc
 some_b = False
 first_b = False
 lwkb = []
 infile = infile_base + '{}_nc_{}_b_{}.npz'.format(res,n_c,0)
 if (os.path.isfile(infile)):
   npz = np.load(infile)
   lwk = npz['Wk_tot']*const
   first_b = True
 for b in range(1,10): 
  infile2 = infile_base + '{}_nc_{}_b_{}.npz'.format(res,n_c,b)
  if (os.path.isfile(infile2)):
    npz = np.load(infile2)
    some_b = True
    lwkb.append(npz['Wk_tot']*const)
 lwkb = np.array(lwkb)
 np.set_printoptions(linewidth=200)


 if (some_b and first_b):
   if (len(lwkb) == 0):
      gap.append(np.nan)
      s_k.append(np.nan)
      wkb.append(np.nan)
   else:
      B = float(len(lwkb))

      # Modified to remove log
      s_k.append(np.sqrt(1 + 1/B) * np.std(lwkb) )

      # Modified to remove log
      wkb.append(np.nanmean(lwkb))
      wkb_sd.append(np.nanstd(lwkb))

      print((n_c,np.nanmean(lwkb) , lwk, gap[-1]))

   n_cs.append(n_c)
   wk.append(lwk)


s_k = np.array(s_k)
n_cs = np.array(n_cs)
wk = np.array(wk)
wkb = np.array(wkb)
wkb_sd = np.array(wkb_sd)

# Smooth things out if many samples and small B
wk = savitzky_golay(wk,3,1)
gap = wkb - wk


fig = plt.figure(facecolor='white',figsize=(8,4))
gs1 = gridspec.GridSpec(1,2)
ax = plt.subplot(gs1[0,0])
plt.errorbar(n_cs[1:],gap[1:],yerr=s_k,c='green')

success = False    
for k in range(1,len(gap)-1):
  if (gap[k] >= gap[k+1] - s_k[k+1]): 
     success = True
     break

if (success == False and np.sum(np.isnan(gap) == False) > 2):
  print('Need more clusters')
else:
  print(('Best cluster at: ',n_cs[k]))


if (os.path.isdir(FIG_DIR) is False):
    os.mkdir(FIG_DIR)
fig.savefig(os.path.join(FIG_DIR,'clustering_curve.png',dpi=100,bbox_inches='tight')


