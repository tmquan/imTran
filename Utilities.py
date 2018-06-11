import numpy as np
import argparse
import os, glob, time
import skimage.io

import PIL
from PIL import Image
import Augmentor
import malis
import malis_loss
from malis_loss import *
from natsort import natsorted
###############################################################################
import tensorflow as tf
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.dataflow import  *              #dataset
from tensorpack.utils import *                  #logger
from tensorpack.utils.gpu import *              #get_nr_gpu
from tensorpack.utils.utils import *            #get_rng
from tensorpack.tfutils import *                #optimizer, gradproc
from tensorpack.tfutils.summary import *        #add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import *    #auto_reuse_variable_scope
###############################################################################
from tensorlayer.cost import * #binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy
###############################################################################

class ClipCallback(Callback):
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()

###############################################################################
# Utility function for scaling 
###############################################################################
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    with tf.variable_scope(name):
        return (x / maxVal - 0.5) * 2.0

def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    with tf.variable_scope(name):
        return (x / 2.0 + 0.5) * maxVal

def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    return (x / maxVal - 0.5) * 2.0

def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    return (x / 2.0 + 0.5) * maxVal


###############################################################################
# Various types of activations
###############################################################################
def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, name=name)
    
def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, name=name)




###############################################################################
def np_seg_to_aff(seg, nhood=malis.mknhood3d(1)):
    # return lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
    seg = np.squeeze(seg)
    seg = seg.astype(np.int32)
    ret = malis.seg_to_affgraph(seg, nhood) # seg zyx
    ret = ret.astype(np.float32)
    ret = np.squeeze(ret) # ret 3zyx
    ret = np.transpose(ret, [1, 2, 3, 0])# ret zyx3
    return ret
def tf_seg_to_aff(seg, nhood=tf.constant(malis.mknhood3d(1)), name='SegToAff'):
    # Squeeze the segmentation to 3D
    seg = tf.cast(seg, tf.int32)
    # Define the numpy function to transform segmentation to affinity graph
    # np_func = lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
    # Convert the numpy function to tensorflow function
    tf_func = tf.py_func(np_seg_to_aff, [seg, nhood], [tf.float32], name=name)
    # Reshape the result, notice that layout format from malis is 3, dimx, dimy, dimx
    # ret = tf.reshape(tf_func[0], [3, seg.shape[0], seg.shape[1], seg.shape[2]])
    # Transpose the result so that the dimension 3 go to the last channel
    # ret = tf.transpose(ret, [1, 2, 3, 0])
    # print seg.get_shape().as_list()
    ret = tf.reshape(tf_func[0], [seg.shape[0], seg.shape[1], seg.shape[2], 3])
    # print ret.get_shape().as_list()
    return ret
###############################################################################
def np_aff_to_seg(aff, nhood=malis.mknhood3d(1), threshold=np.array([0.5]) ):
    aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
    ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
    ret = skimage.measure.label(ret).astype(np.float32)
    return ret
def tf_aff_to_seg(aff, nhood=tf.constant(malis.mknhood3d(1)), threshold=tf.constant(np.array([0.5])), name='AffToSeg'):
    # Define the numpy function to transform affinity to segmentation
    # def np_func (aff, nhood, threshold):
    #   aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
    #   ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
    #   ret = skimage.measure.label(ret).astype(np.int32)
    #   return ret
    # print aff.get_shape().as_list()
    # Convert numpy function to tensorflow function
    tf_func = tf.py_func(np_aff_to_seg, [aff, nhood, threshold], [tf.float32], name=name)
    ret = tf.reshape(tf_func[0], [aff.shape[0], aff.shape[1], aff.shape[2]])
    ret = tf.expand_dims(ret, axis=-1)
    # print ret.get_shape().as_list()
    return ret



