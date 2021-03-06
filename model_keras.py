from config import config
import tensorflow as tf
import numpy as np

class KPLayer(tf.keras.layers.Layer):
    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self._kp_kernel_conv = None
        self._kp_bias = None
        self._debug = debug

    def build(self, input_shape):
        if self._kp_kernel_conv is None:
            self._kp_kernel_conv = tf.Variable(np.zeros((1,1,2048,config.NUM_KP)), dtype=tf.float32, name='KPLayer_kernel')
        if self._kp_bias is None:
            self._kp_bias = tf.Variable(np.zeros((1,1,config.NUM_KP)), dtype=tf.float32, name='KPLayer_bias')

    def call(self, x):        
        z = tf.nn.conv2d(x, self._kp_kernel_conv, strides=[1, 1, 1, 1], padding='VALID')
        z = z + self._kp_bias
        z = tf.math.sigmoid(z)
        if self._debug:
            tf.Print(f'KP output: {z}')

        return z

class ShortOffsetLayer(tf.keras.layers.Layer):
    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self._kp_kernel_conv = None
        self._kp_bias = None
        self._debug = debug

    def build(self, input_shape):
        if self._kp_kernel_conv is None:
            self._kp_kernel_conv = tf.Variable(np.zeros((1,1,2048, 2 * config.NUM_KP)), dtype=tf.float32, name='ShortOffsetLayer_kernel')
        if self._kp_bias is None:
            self._kp_bias = tf.Variable(np.zeros((1,1,2 * config.NUM_KP)), dtype=tf.float32, name='ShortOffsetLayer_bias')
        
    def call(self, x):
        z = tf.nn.conv2d(x, self._kp_kernel_conv, strides=[1, 1, 1, 1], padding='VALID')
        z = z + self._kp_bias
        if self._debug:
            tf.Print(f'Short Offset output: {z}')

        return z

class MiddleOffsetLayer(tf.keras.layers.Layer):
    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self._kp_kernel_conv = None
        self._kp_bias = None
        self._debug = debug

    def build(self, input_shape):
        if self._kp_kernel_conv is None:
            self._kp_kernel_conv = tf.Variable(np.zeros((1,1,2048, 4 * config.NUM_EDGES)), dtype=tf.float32, name='MiddleOffsetLayer_kernel')
        if self._kp_bias is None:
            self._kp_bias = tf.Variable(np.zeros((1,1,4 * config.NUM_EDGES)), dtype=tf.float32, name='MiddleOffsetLayer_bias')
        
    def call(self, x):        
        z = tf.nn.conv2d(x, self._kp_kernel_conv, strides=[1, 1, 1, 1], padding='VALID')
        z = z + self._kp_bias
        if self._debug:
            tf.Print(f'Middle Offset output: {z}')

        return z

class LongOffsetLayer(tf.keras.layers.Layer):
    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self._kp_kernel_conv = None
        self._kp_bias = None
        self._debug = debug
    
    def build(self, input_shape):
        if self._kp_kernel_conv is None:
            self._kp_kernel_conv = tf.Variable(np.zeros((1,1,2048, 2*config.NUM_KP)), dtype=tf.float32, name='LongOffsetLayer_kernel')
        if self._kp_bias is None:
            self._kp_bias = tf.Variable(np.zeros((1,1,2*config.NUM_KP)), dtype=tf.float32,  name='LongOffsetLayer_bias')

    def call(self, x):        
        z = tf.nn.conv2d(x, self._kp_kernel_conv, strides=[1, 1, 1, 1], padding='VALID')
        z = z + self._kp_bias
        if self._debug:
            tf.Print(f'Long Offset output: {z}')

        return z

class SemSegLayer(tf.keras.layers.Layer):
    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self._kp_kernel_conv = None
        self._kp_bias = None
        self._debug = debug

    def build(self, input_shape):
        if self._kp_kernel_conv is None:
            self._kp_kernel_conv = tf.Variable(np.zeros((1,1,2048, 1)), dtype=tf.float32, name='SemSegLayer_kernel')
        if self._kp_bias is None:
            self._kp_bias = tf.Variable(np.zeros((1,1,1)), dtype=tf.float32, name='SemSegLayer_bias')

    def call(self, x):        
        z = tf.nn.conv2d(x, self._kp_kernel_conv, strides=[1, 1, 1, 1], padding='VALID')
        z = z + self._kp_bias
        z = tf.math.sigmoid(z)

        if self._debug:
            tf.Print(f'Sem Seg output: {z}')

        return z

def bilinear_sampler(x, v):

  def _get_grid_array(N, H, W, h, w):
    N_i = tf.range(N)
    H_i = tf.range(h+1, h+H+1)
    W_i = tf.range(w+1, w+W+1)
    n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
    h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
    w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
    n = tf.cast(n, tf.float32) # [N, H, W, 1]
    h = tf.cast(h, tf.float32) # [N, H, W, 1]
    w = tf.cast(w, tf.float32) # [N, H, W, 1]

    return n, h, w

  shape = tf.shape(x) # TRY : Dynamic shape
  N = shape[0]
  H_ = H = shape[1]
  W_ = W = shape[2]
  h = w = 0

  
  x = tf.pad(x,
    ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
  
  vx, vy = tf.split(v, 2, axis=3)
  

  n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]

  vx0 = tf.math.floor(vx)
  vy0 = tf.math.floor(vy)
  vx1 = tf.math.ceil(vx)
  vy1 = tf.math.ceil(vy) # [N, H, W, 1]

  iy0 = vy0 + h
  iy1 = vy1 + h
  ix0 = vx0 + w
  ix1 = vx1 + w

  H_f = tf.cast(H_, tf.float32)
  W_f = tf.cast(W_, tf.float32)
  mask = tf.math.less(ix0, 1)
  mask = tf.math.logical_or(mask, tf.math.less(iy0, 1))
  mask = tf.math.logical_or(mask, tf.math.greater(ix1, W_f))
  mask = tf.math.logical_or(mask, tf.math.greater(iy1, H_f))

  iy0 = tf.where(mask, tf.zeros_like(iy0), iy0)
  iy1 = tf.where(mask, tf.zeros_like(iy1), iy1)
  ix0 = tf.where(mask, tf.zeros_like(ix0), ix0)
  ix1 = tf.where(mask, tf.zeros_like(ix1), ix1)


  i00 = tf.concat([n, iy0, ix0], 3)
  i01 = tf.concat([n, iy1, ix0], 3)
  i10 = tf.concat([n, iy0, ix1], 3)
  i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
  i00 = tf.cast(i00, tf.int32)
  i01 = tf.cast(i01, tf.int32)
  i10 = tf.cast(i10, tf.int32)
  i11 = tf.cast(i11, tf.int32)

  x00 = tf.gather_nd(x, i00)
  x01 = tf.gather_nd(x, i01)
  x10 = tf.gather_nd(x, i10)
  x11 = tf.gather_nd(x, i11)

  dx = tf.cast(vx - vx0, tf.float32)
  dy = tf.cast(vy - vy0, tf.float32)
  
  w00 = (1.-dx) * (1.-dy)
  w01 = (1.-dx) * dy
  w10 = dx * (1.-dy)
  w11 = dx * dy
  
  output = tf.math.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

  return output

def refine(base,offsets,num_steps=2):
    for i in range(num_steps):
        base = base + bilinear_sampler(offsets,base)
    return base

def split_and_refine_mid_offsets(mid_offsets, short_offsets):
    output_mid_offsets = []
    for mid_idx, edge in enumerate(config.EDGES+[edge[::-1] for edge in config.EDGES]):
        to_keypoint = edge[1]
        kp_short_offsets = short_offsets[:,:,:,2*to_keypoint:2*to_keypoint+2]
        kp_mid_offsets = mid_offsets[:,:,:,2*mid_idx:2*mid_idx+2]
        kp_mid_offsets = refine(kp_mid_offsets,kp_short_offsets,2)
        output_mid_offsets.append(kp_mid_offsets)
    return tf.concat(output_mid_offsets,axis=-1)

def split_and_refine_long_offsets(long_offsets, short_offsets):
    output_long_offsets = []
    for i in range(config.NUM_KP):
        kp_long_offsets = long_offsets[:,:,:,2*i:2*i+2]
        kp_short_offsets = short_offsets[:,:,:,2*i:2*i+2]
        refine_1 = refine(kp_long_offsets,kp_long_offsets)
        refine_2 = refine(refine_1,kp_short_offsets)
        output_long_offsets.append(refine_2)
    return tf.concat(output_long_offsets,axis=-1)

class Net(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self._nm = tf.keras.layers.experimental.preprocessing.Normalization()
        self._nm_mean = np.array([127.5] * 3)
        self._nm_var = self._nm_mean ** 2
        self._resnet = tf.keras.applications.ResNet101V2(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=(401, 401, 3),
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        self._kp = KPLayer()
        self._short_offset = ShortOffsetLayer()
        self._middle_offset = MiddleOffsetLayer()
        self._long_offset = LongOffsetLayer()
        self._semseg = SemSegLayer()
        self._resnet.trainable = False
            
    def call(self, x):
        result = []
        x = self._nm(x)
        x = self._resnet(x)
        kp_map = self._kp(x)
        short_offset_map = self._short_offset(x)
        middle_offset_map = self._middle_offset(x)
        long_offset_map = self._long_offset(x)
        semseg_map = self._semseg(x)
        
        # resize
        kp_map = tf.image.resize(kp_map, 
                        (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                        method=tf.image.ResizeMethod.BILINEAR, 
                        preserve_aspect_ratio=False,
                        antialias=False, 
                        name=None)
        short_offset_map = tf.image.resize(short_offset_map, 
                        (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                        method=tf.image.ResizeMethod.BILINEAR, 
                        preserve_aspect_ratio=False,
                        antialias=False, 
                        name=None)
        middle_offset_map = tf.image.resize(middle_offset_map, 
                        (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                        method=tf.image.ResizeMethod.BILINEAR, 
                        preserve_aspect_ratio=False,
                        antialias=False, 
                        name=None)
        long_offset_map = tf.image.resize(long_offset_map, 
                        (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                        method=tf.image.ResizeMethod.BILINEAR, 
                        preserve_aspect_ratio=False,
                        antialias=False, 
                        name=None)
        semseg_map = tf.image.resize(semseg_map, 
                        (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                        method=tf.image.ResizeMethod.BILINEAR, 
                        preserve_aspect_ratio=False,
                        antialias=False, 
                        name=None)

        mid_offsets = split_and_refine_mid_offsets(middle_offset_map, short_offset_map)
        long_offsets = split_and_refine_long_offsets(long_offset_map, short_offset_map)

        result.append(kp_map)
        result.append(short_offset_map)
        result.append(mid_offsets)
        result.append(long_offsets)
        result.append(semseg_map)

        return result

