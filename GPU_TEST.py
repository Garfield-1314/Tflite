import tensorflow as tf

# 检查 TensorFlow 是否可以访问 GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("CUDA GPU 可用！")
    # 打印 GPU 信息
    for gpu in gpus:
        print(f"GPU 名称: {gpu.name}")
else:
    print("没有检测到 GPU，可能没有启用 CUDA。")

# 检查 TensorFlow 是否检测到 CUDA 设备
cuda_available = tf.test.is_built_with_cuda()
if cuda_available:
    print("TensorFlow 是基于 CUDA 构建的。")
else:
    print("TensorFlow 并未基于 CUDA 构建。")
