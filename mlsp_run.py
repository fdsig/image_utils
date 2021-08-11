import kutils
from kutils import model_helper as mh
from kutils import applications as apps
from kutils import tensor_ops as ops
from kutils import generic as gen


import pandas as pd, numpy as np, os
root_path = '/content/ava-mlsp/'
dataset = root_path + 'metadata/AVA_data_official_test.csv';
ids = pd.read_csv(dataset)

features_file = root_path + 'models/irnv2_mlsp_wide_orig/model_best_weights.h5'
fc1_size = 2048
image_size = '[orig]'
input_size = (5,5,16928)
model_name = features_file.split('/')[-2]
loss = 'MSE'
bn = 2
fc_sizes = [fc1_size, fc1_size/2, fc1_size/8,  1]
dropout_rates = [0.25, 0.25, 0.5, 0]

monitor_metric = 'val_plcc_tf'; monitor_mode = 'max'
metrics = ["MAE", ops.plcc_tf]
outputs = 'MOS'

# MODEL DEF
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
import keras

input_feats = Input(shape=input_size, dtype='float32')

# SINGLE-block
x = apps.inception_block(input_feats, size=1024)
x = GlobalAveragePooling2D(name='final_GAP')(x)

pred = apps.fc_layers(x, name       = 'head',
                 fc_sizes      = fc_sizes,
                 dropout_rates = dropout_rates,
                 batch_norm    = bn)

model = Model(inputs=input_feats, outputs=pred)

gen_params = dict(batch_size    = 128,
                  data_path     = root_path+'images/',                  
                  input_shape   = input_size,
                  inputs        = 'image_name',
                  outputs       = outputs, 
                  random_group  = False,
                  fixed_batches = True)

helper = mh.ModelHelper(model, model_name, ids, 
                     max_queue_size = 128,
                     loss           = loss,
                     metrics        = metrics,
                     monitor_metric = monitor_metric, 
                     monitor_mode   = monitor_mode,
                     multiproc      = False, workers = 1,
#                      multiproc      = True, workers = 3,
                     early_stop_patience = 5,
                     logs_root      = root_path + 'logs',
                     models_root    = root_path + 'models',
                     gen_params     = gen_params)

helper.model_name.update(fc1 = '[%d]' % fc1_size, 
                         im  = image_size,
                         bn  = bn,
                         do  = str(dropout_rates).replace(' ',''),
                         mon = '[%s]' % monitor_metric,
                         ds  = '[%s]' % os.path.split(dataset)[1])

print helper.model_name()

for lr in [1e-4,1e-5,1e-6]:
    helper.load_model()
    helper.train(lr=lr, epochs=20)
