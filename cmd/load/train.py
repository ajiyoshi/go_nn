# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

import msgpack
import msgpack_numpy as m
m.patch()

def save(path, nn):
    grads = {}
    grads['W1'], grads['b1'] = nn.layers['Conv1'].dW, nn.layers['Conv1'].db
    grads['W2'], grads['b2'] = nn.layers['Affine1'].dW, nn.layers['Affine1'].db
    grads['W3'], grads['b3'] = nn.layers['Affine2'].dW, nn.layers['Affine2'].db
    with open(path, 'wb') as f:
        f.write(msgpack.packb(grads, default=m.encode))

def save_array(path, array):
    with open(path, 'wb') as f:
        f.write(msgpack.packb(array, default=m.encode))

def save_layer(path, layer):
    obj = {}
    obj['W'] = layer.W
    obj['dW'] = layer.dW
    obj['db'] = layer.db
    with open(path, 'wb') as f:
        f.write(msgpack.packb(obj, default=m.encode))


# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=3,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1)

#trainer.train_step()

batch_mask = np.array([0, 1, 2])
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

conv = network.layers['Conv1']

save_array("full.x.mp", x_batch)
save_array("full.W0.mp", conv.W)

grads = network.gradient(x_batch, t_batch)
trainer.optimizer.update(trainer.network.params, grads)

save_array("full.W1.mp", conv.W)
save_array("full.b1.mp", conv.b)
