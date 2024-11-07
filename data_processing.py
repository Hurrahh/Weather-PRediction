import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

train_dir = 'weather dataset/train'
test_dir = 'weather dataset/test'


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array /= 255.0
    return img_array

image_size = (128, 128)
def load_images_from_directory(directory):
    images = []
    labels = []
    label_map = {'cloud': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}

    for label_name, label in label_map.items():
        label_folder = os.path.join(directory, label_name)
        for filename in os.listdir(label_folder):
            image_path = os.path.join(label_folder, filename)
            img = load_and_preprocess_image(image_path, target_size=image_size)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


train_images, train_labels = load_images_from_directory(train_dir)
test_images, test_labels = load_images_from_directory(test_dir)

train_labels = to_categorical(train_labels, num_classes=4)
test_labels = to_categorical(test_labels, num_classes=4)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def visualize_sample_images(images, labels, class_names, num_samples=5):
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.show()

class_names = ['Cloud', 'Rain', 'Shine', 'Sunrise']

visualize_sample_images(train_images, train_labels, class_names, num_samples=5)

plt.figure(figsize=(6, 4))
sns.countplot(x=np.argmax(train_labels, axis=1))
plt.xticks(ticks=np.arange(4), labels=class_names)
plt.title('Class Distribution in Training Dataset')
plt.ylabel('Count')
plt.xlabel('Weather Class')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_images_grid(images, labels, class_names, n_samples=5):
    plt.figure(figsize=(15, 12))
    for idx, label in enumerate(np.unique(np.argmax(labels, axis=1))):
        class_images = images[np.argmax(labels, axis=1) == label][:n_samples]
        for i, img in enumerate(class_images):
            plt.subplot(len(np.unique(np.argmax(labels, axis=1))), n_samples, idx * n_samples + i + 1)
            plt.imshow(img)
            plt.title(class_names[label])
            plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_images_grid(train_images, train_labels, class_names, n_samples=5)


def plot_image_dimensions(images):
    image_heights = [img.shape[0] for img in images]
    image_widths = [img.shape[1] for img in images]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(image_heights, bins=50, color='skyblue', edgecolor='black')
    plt.title('Image Height Distribution')
    plt.xlabel('Height')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(image_widths, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Image Width Distribution')
    plt.xlabel('Width')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_image_dimensions(train_images)


def plot_rgb_histograms(images, class_labels, class_names):
    plt.figure(figsize=(15, 10))

    for idx, label in enumerate(np.unique(np.argmax(class_labels, axis=1))):
        class_images = images[np.argmax(class_labels, axis=1) == label]

        reds = []
        greens = []
        blues = []

        for img in class_images:
            reds.extend(img[:, :, 0].flatten())
            greens.extend(img[:, :, 1].flatten())
            blues.extend(img[:, :, 2].flatten())

        plt.subplot(2, 2, idx + 1)
        plt.hist(reds, bins=50, color='red', alpha=0.6, label='Red')
        plt.hist(greens, bins=50, color='green', alpha=0.6, label='Green')
        plt.hist(blues, bins=50, color='blue', alpha=0.6, label='Blue')

        plt.title(f'RGB Distribution - {class_names[label]}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()

plot_rgb_histograms(train_images, train_labels, class_names)
