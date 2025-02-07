
import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore
import matplotlib.pyplot as plt
import numpy as np
 
# 导入数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 观察数据
print (x_train.shape)
plt.imshow(x_train[1000])
print (y_train[1000])
 
train_images=x_train/255.0
 
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0
 
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
 
plt.imshow(x_train[2000])
 
 
x_train = x_train.reshape((x_train.shape[0],28,28,1)).astype('float32') 
x_test = x_test.reshape((x_test.shape[0],28,28,1)).astype('float32') #-1代表那个地方由其余几个值算来的
x_train = x_train/255
x_test = x_test/255
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
print (x_train.shape)