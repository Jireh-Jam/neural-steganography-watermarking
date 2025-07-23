import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
def build_encoder():
    cover_input = layers.Input(shape=(128, 128, 3), name="cover_input")
    message_input = layers.Input(shape=(128, 128, 1), name="message_input")
    x = layers.Concatenate()([cover_input, message_input])
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    stego_output = layers.Conv2D(3, (1,1), activation='sigmoid', name="stego_output")(x)
    return keras.Model([cover_input, message_input], stego_output)
