import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

base_dir = './dataset'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
# test_dir = os.path.join(base_dir, 'test')
# test2_dir = os.path.join(base_dir, 'test2')



BATCH_SIZE = 128
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            validation_split=0.2,
                                                            subset="training",
                                                            seed=12,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(valid_dir,
                                                                 validation_split=0.2,
                                                                 subset="validation",
                                                                 seed=12,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

print(train_dataset.class_names)
class_len=len(train_dataset.class_names)
print(class_len)


plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_dataset.class_names[labels[i]])
    plt.axis("off")


IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               alpha=0.25,
                                               weights='imagenet')
base_model.trainable = True
base_model.summary()

normalization_layer = tf.keras.layers.Rescaling(1. / 127.5,input_shape=(160, 160, 3)) 
normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

model = tf.keras.Sequential([
  normalization_layer,
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomBrightness(0.3),
  tf.keras.layers.RandomContrast(0.3),
  tf.keras.layers.RandomHeight(0.3),
  tf.keras.layers.RandomRotation(1,seed=12),
  tf.keras.layers.RandomZoom(height_factor=0.4,fill_value=0.3),
  base_model,
  # tf.keras.layers.Conv2D(64, 3),
  # tf.keras.layers.Conv2D(32, 32),
  # tf.keras.layers.Dense(32, activation='relu'),
  # tf.keras.layers.BatchNormalization(),
  # tf.keras.layers.Conv2D(64, 3, activation='relu'),
  # tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(class_len,activation='softmax')
])

base_learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

initial_epochs = 100

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=initial_epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100000):
    yield [tf.dtypes.cast(data, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model_quant = converter.convert()


# 保存模型
# model.save('./model/orginal.h5') # 保存原始模型

open('./model/mnist_model.tflite', "wb").write(tflite_model) # 保存tflite-float32原始模型

open('./model/mnist_model_quant.tflite', "wb").write(tflite_model_quant) # 保存tflite-int8量化模型

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()