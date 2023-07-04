import os
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load Image ------------------------------------------------------------
Abnormal_list = os.listdir('./image/abnormal')
Normal_list = os.listdir('./image/normal')

num_Abnormal_list = len(Abnormal_list)
num_Normal_list = len(Normal_list)
num_Total_list = num_Abnormal_list + num_Normal_list


# Image PreProcessing ---------------------------------------------------
num = 0
all_img = np.float32(np.zeros((num_Total_list, 224, 224, 3)))
all_label = np.float64(np.zeros((num_Total_list, 1)))

for i in Abnormal_list:
    img_path = './image/abnormal/' + i
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    all_label[num] = 0
    num = num + 1

for i in Normal_list:
    img_path = './image/normal/' + i
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    all_label[num] = 1
    num = num + 1

# Shuffle Image Data
n_element = all_label.shape[0]
indices = np.random.choice(n_element, size=n_element, replace=False)

all_label = all_label[indices]
all_img = all_img[indices]

# Split Images to Training Set/Test Set
num_train = int(np.round(all_label.shape[0] * 0.8))
num_test = int(np.round(all_label.shape[0] * 0.2))

train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :]

train_label = all_label[0:num_train]
test_label = all_label[num_train:]


# Create Trained Model --------------------------------------------------
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)

model = Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    bn_layer1,
    dense_layer2,
])

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_img, train_label, epochs=10, batch_size=16, validation_data=(test_img, test_label))

# Save Model
model.save("model.h5")

print("Saved model to disk")