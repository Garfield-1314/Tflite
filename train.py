# Jupyter Notebook - 代码

# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设定日志级别，减少不必要的输出
tf.get_logger().setLevel('ERROR')


# 🔹 数据集路径
base_dir = './dataset'
train_dir = os.path.join(base_dir, 'YASUO_80')
valid_dir = os.path.join(base_dir, 'YASUO_80')

# 🔹 超参数
BATCH_SIZE = 4
IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE

# 🔹 加载数据集
train_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="training", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)  # ✅ batch_size 已经指定

validation_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir, validation_split=0.2, subset="validation", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)  # ✅ batch_size 已经指定

class_names = train_dataset_raw.class_names
print("Class Names:", class_names)

# 🔹 数据增强
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.3)
])

# 预处理函数
def preprocess_image(image, label):
    image = data_augmentation(image)  # 添加数据增强
    # image = tf.image.convert_image_dtype(image, tf.float32)  # 归一化
    return image, label

# 加载数据集 & 预处理
train_dataset = (train_dataset_raw
                 .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                 .cache()
                 .shuffle(1000)
                 .prefetch(AUTOTUNE))  # ✅ 不再 batch()

validation_dataset = (validation_dataset_raw
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                      .cache()
                      .prefetch(AUTOTUNE))  # ✅ 不再 batch()

# 🔹 预训练模型（MobileNet）
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNet(
    input_shape=IMG_SHAPE, include_top=False, alpha=0.25, weights='imagenet')

# 仅微调最后4层
base_model.trainable = True
for layer in base_model.layers[:-0]:
    layer.trainable = False

base_model.summary()
# 🔹 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),  # 归一化到 [0,1]
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),  # Dropout 防止过拟合
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # 分类层
])
model.build((None, 160, 160, 3))
model.summary()

# 🔹 指数学习率衰减
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.90, staircase=True)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 训练回调（去掉 ReduceLROnPlateau）
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# 训练模型
history = model.fit(train_dataset, validation_data=validation_dataset,
                    epochs=25, callbacks=[early_stopping])

# 🔹 代表性数据集（INT8 量化）
def representative_dataset():
    for image_batch, _ in train_dataset.take(5000):
        yield [tf.cast(image_batch[0:1], tf.float32)]  # 仅返回单张图片

# 🔹 量化模型为 TFLite INT8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

# 🔹 保存 TFLite 量化模型
os.makedirs('./model', exist_ok=True)
with open('./model/mnist_model_quant_uint8.tflite', "wb") as f:
    f.write(tflite_model_quant)

# 🔹 推理测试（检查 TFLite 输入输出类型）
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
print('Input Type:', interpreter.get_input_details()[0]['dtype'])
print('Output Type:', interpreter.get_output_details()[0]['dtype'])

# 🔹 训练曲线可视化
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()

# 🔹 混淆矩阵（分析分类错误）
y_pred = np.argmax(model.predict(validation_dataset), axis=1)
y_true = np.concatenate([labels.numpy() for _, labels in validation_dataset])

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
