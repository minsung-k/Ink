# https://keras.io/examples/vision/grad_cam/

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------

def heatmap(img_array, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the result for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)


    # This is the gradient of the output neuron
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(preds, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = heatmap.numpy()
    #preprocessing
    heatmap[:,0] = 0
    heatmap[:,-1] = 0
    heatmap[-1,:] = 0
    heatmap = np.where(heatmap< 0.2, 0,heatmap)
    
    return heatmap

# ---------------------------------------------------------------------------------


def display(img, heatmap):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap + img * 255
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))
    
    #plt.imshow(superimposed_img)
    
    return superimposed_img

# ---------------------------------------------------------------------------------

def collect_heatmap(X,model):
    list_heatmap = []
    
    for i in range(len(X)):
        image = np.expand_dims(X[i], axis=0)
        hp = heatmap(image, model, "block4_pool")
        list_heatmap.append(hp)
        
    return list_heatmap


# ---------------------------------------------------------------------------------

def collect_ewma(list_heatmap, alpha):
    
    list_ewma = []

    for i in range(len(list_heatmap)):
        
        previous_result = np.zeros([43,6])
        after_result = np.zeros([43,6])

        num = 1
        p_alpha = alpha
        while True:

            if i - num >=0:

                result = list_heatmap[i-num] * p_alpha
                previous_result = previous_result+ result

                num = num+1
                p_alpha = p_alpha * p_alpha

            else: break

        num = 1
        a_alpha = alpha

        while True:

            try:
                result = list_heatmap[i+num] * a_alpha
                after_result = after_result+ result

                num = num+1
                a_alpha = a_alpha * a_alpha

            except:
                break

        current_result = list_heatmap[i] * (1-alpha)

        heatmaps = previous_result + after_result + current_result
        list_ewma.append(heatmaps)
    return list_ewma
        
# ---------------------------------------------------------------------------------

def collect_superimposed(list_ewma, X):
    list_superimposed = []
    
    for i in range(len(X)):
        
        heatmap = np.uint8(255 * list_ewma[i])
        
        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((X[i].shape[1], X[i].shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap + X[i] * 255
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
        
        list_superimposed.append(superimposed_img)
        
    return list_superimposed
# ---------------------------------------------------------------------------------