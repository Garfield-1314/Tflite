import tensorflow as tf

# 打印 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 检测是否使用了 GPU
gpu_available = tf.test.is_gpu_available()
print("GPU 是否可用:", gpu_available)

# 打印 CUDA 版本
if gpu_available:
    print("CUDA 版本:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN 版本:", tf.sysconfig.get_build_info()["cudnn_version"])
else:
    print("CUDA 版本: 未检测到 GPU")