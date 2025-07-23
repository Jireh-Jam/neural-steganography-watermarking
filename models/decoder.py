import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def build_decoder():
    stego_input = layers.Input(shape=(128, 128, 3), name="stego_input")
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(stego_input)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    message_output = layers.Conv2D(1, (1,1), activation='sigmoid', name="message_output")(x)
    return keras.Model(stego_input, message_output)
