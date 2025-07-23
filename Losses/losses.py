import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def build_perceptual_loss_model():
    # Load pre-trained VGG16 without top layers
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # Extract features from specific layers
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3'] 
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    return Model(inputs=vgg.input, outputs=outputs)

perceptual_model = build_perceptual_loss_model()
perceptual_model.trainable = False  # Freeze VGG weights

def build_feature_extractor(model_name='vgg19'):
    if model_name == 'vgg19':
        base = VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))
        layers = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']
    elif model_name == 'mobilenetv2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
        layers = ['block_2_expand_relu', 'block_5_expand_relu', 'block_12_expand_relu']
    elif model_name == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(128,128,3))
        layers = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
    
    outputs = [base.get_layer(name).output for name in layers]
    return Model(inputs=base.input, outputs=outputs), len(layers)

def perceptual_loss(y_true, y_pred, model_name='vgg19'):
    extractor, num_layers = build_feature_extractor(model_name)
    extractor.trainable = False
    
    # Normalize images for the specific model
    if model_name == 'vgg19':
        y_true = tf.keras.applications.vgg19.preprocess_input(y_true * 255.)
        y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.)
    else:
        y_true = tf.keras.applications.mobilenet_v2.preprocess_input(y_true * 255.)
        y_pred = tf.keras.applications.mobilenet_v2.preprocess_input(y_pred * 255.)
    
    # Get features and calculate loss
    true_features = extractor(y_true)
    pred_features = extractor(y_pred)
    
    loss = 0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.square(true_feat - pred_feat))
    return loss / num_layers  # Normalize by number of layers

def compute_vgg_loss(original, generated, vgg_model):
    # Preprocess images (scale to 0-255 and apply model-specific preprocessing)
    orig_processed = tf.keras.applications.vgg19.preprocess_input(original * 255.)
    gen_processed = tf.keras.applications.vgg19.preprocess_input(generated * 255.)
    
    # Get features from multiple layers
    orig_features = vgg_model(orig_processed)
    gen_features = vgg_model(gen_processed)
    
    # Calculate MSE between feature maps
    loss = sum(tf.reduce_mean(tf.square(o - g)) 
               for o, g in zip(orig_features, gen_features))
    return loss / vgg_model.num_layers

def build_combined_loss(vgg_model):
    """Creates a combined loss function with VGG, SSIM, and message components."""
    def combined_loss(cover, stego, msg_loss):
        # Perceptual loss
        vgg_loss = compute_vgg_loss(cover, stego, vgg_model)
        
        # Structural similarity loss
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(cover, stego, max_val=1.0))
        
        return 0.6*vgg_loss + 0.3*ssim_loss + 0.1*msg_loss
    return combined_loss
