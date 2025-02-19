import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # 导入 tqdm 库

# 加载量化后的TFLite模型
interpreter = tf.lite.Interpreter(model_path='./model/mnist_model_quant_uint8.tflite')
interpreter.allocate_tensors()

# 获取输入和输出的张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 数据集路径
base_dir = './dataset'
valid_dir = os.path.join(base_dir, 'Origin')

# 超参数设置
BATCH_SIZE = 512
IMG_SIZE = (128, 128)

# 加载验证数据集
validation_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(valid_dir,
                                                                             validation_split=0.99,
                                                                             subset="validation",
                                                                             seed=12,
                                                                             batch_size=BATCH_SIZE,
                                                                             image_size=IMG_SIZE)

# 获取类名
class_names = validation_dataset_raw.class_names

def predict(image):
    image = tf.expand_dims(image, axis=0)  # 扩展维度为 (1, 160, 160, 3)
    input_data = np.array(image, dtype=np.uint8)  # 输入数据转为 INT8
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# 计算每类的准确率
num_classes = len(class_names)
correct_predictions = np.zeros(num_classes)
total_predictions = np.zeros(num_classes)

# 遍历验证集进行推理
for images, labels in tqdm(validation_dataset_raw, desc="Processing batches"):  # 添加进度条
    for image, label in zip(images, labels):
        predictions = predict(image)
        predicted_label = np.argmax(predictions, axis=1)[0]

        total_predictions[label.numpy()] += 1
        if label.numpy() == predicted_label:
            correct_predictions[label.numpy()] += 1

# 计算每个类别的准确率，并转换为百分比
class_accuracies = correct_predictions / total_predictions
class_accuracies_percentage = class_accuracies * 100  # 转换为百分比

# 绘制准确率柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_accuracies_percentage, color='skyblue')

# 在柱子顶部显示数值
for bar, accuracy in zip(bars, class_accuracies_percentage):
    plt.text(bar.get_x() + bar.get_width()/2,  # 柱子的中心位置
             bar.get_height(),  # 柱子的高度
             f'{accuracy:.1f}%',  # 以百分比格式显示
             ha='center',  # 居中对齐
             va='bottom',  # 位置在柱子顶部
             fontsize=10)

plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Class-wise Accuracy')
plt.xticks(rotation=90)
plt.ylim(0, 100)  # 让 y 轴范围为 0-100
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()