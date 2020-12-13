from converter import converter
from makebatches import *
from rbm import rbm
from backprop import backprop
from rbmhidlinear import rbmhidlinear
import scipy.io as sio
from mnistdisp import mnistdisp

MAX_EPOCH = 10
NUM_HID = 1000
NUMPEN = 500
NUMPEN2 = 250
NUMOPEN = 30

# print('Converting Raw files into Matlab format')
# converter()

# print('Pretraining a deep autoencoder.')
# print('The Science paper used 50 epochs. This uses', MAX_EPOCH)

# BATCH_DATA = makebatches()
RANDOMIN_ = sio.loadmat("randomin_poshidstates.mat", verify_compressed_data_integrity=False)
RANDOMIN = RANDOMIN_['randomin']
VISHID_ = sio.loadmat("vis_try___.mat", verify_compressed_data_integrity=False)
VISHID = VISHID_['vishid_']

BATCH_DATA_ = sio.loadmat("batchdata_py.mat", verify_compressed_data_integrity=False)
BATCH_DATA = BATCH_DATA_['batchdata']
NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape
# print("BATCH_DATA.shape= ", BATCH_DATA.shape) #100, 784, 600

print('Pretraining Layer 1 with RBM: {}-{}'.format(NUM_DIMS, NUM_HID))
RESTART = 1
BATCHPOSHIDPROBS, VISHID, HIDBIASES, VISBIASES = rbm(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH, RANDOMIN, VISHID)
print("BATCHPOSHIDPROBS.shape= ", BATCHPOSHIDPROBS.shape) #100, 1000
HIDRECBIASES = HIDBIASES
VISHID__= VISHID
VISBIASES__ = VISBIASES

print('Pretraining Layer 2 with RBM: {}-{}'.format(NUM_HID, NUMPEN))
BATCH_DATA = BATCHPOSHIDPROBS
NUM_HID = NUMPEN
RESTART = 1
BATCHPOSHIDPROBS, VISHID, HIDBIASES, VISBIASES = rbm(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH, RANDOMIN, VISHID)
HIDPEN = VISHID
PENRECBIASES = HIDBIASES
HIDGENBIASES = VISBIASES

print('Pretraining Layer 3 with RBM: {}-{}'.format(NUMPEN, NUMPEN2))
BATCH_DATA = BATCHPOSHIDPROBS
NUM_HID = NUMPEN2
RESTART = 1
BATCHPOSHIDPROBS, VISHID, HIDBIASES, VISBIASES = rbm(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH, RANDOMIN, VISHID)
HIDPEN2 = VISHID
PENRECBIASES2 = HIDBIASES
HIDGENBIASES2 = VISBIASES

print('Pretraining Layer 4 with RBM: {}-{}'.format(NUMPEN2, NUMOPEN))
BATCH_DATA = BATCHPOSHIDPROBS
NUM_HID = NUMOPEN
RESTART = 1
VISHID, HIDBIASES, VISBIASES = rbmhidlinear(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH, RANDOMIN, VISHID)
HIDTOP = VISHID
TOPRECBIASES = HIDBIASES
TOPGENBIASES = VISBIASES

print('END of Pretraining Layer 4 with RBM: {}-{}'.format(NUMPEN2, NUMOPEN))

ERR = backprop(VISHID__, VISBIASES__, PENRECBIASES, PENRECBIASES2, HIDRECBIASES, HIDPEN, HIDPEN2, HIDGENBIASES, HIDGENBIASES2, HIDTOP, TOPRECBIASES, TOPGENBIASES)

