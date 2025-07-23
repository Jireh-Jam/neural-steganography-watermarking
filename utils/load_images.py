import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


def load_image(path, target_size=(128, 128)):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    return tf.cast(img, tf.float32) / 255.0


def create_folder(folder_name, base_path=None):
    if base_path is None:
        base_path = os.getcwd()
    full_path = os.path.join(base_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def load_all_imgs(path, size=(128,128)):
    imgs = []
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
            full_path = os.path.join(path, filename)
            img = Image.open(full_path).convert('RGB').resize(size)
            imgs.append(np.asarray(img)/255.0)
    return np.stack(imgs)
