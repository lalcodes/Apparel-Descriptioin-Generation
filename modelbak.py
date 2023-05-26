import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
# import collections
# import random
import numpy as np
import os
# import time
# import json
from PIL import Image
# import glob
#from IPython.display import display, Image
import urllib.request
from nanoid import generate
import boto3
import key_config as keys
import re

data = pd.read_csv('data/FinalData_multipose_v1.csv')

captions = data.captions

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

caption_dataset = tf.data.Dataset.from_tensor_slices(captions)

def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

embedding_dim = 256
units = 512
# Max word count for a caption.
max_length = 20
# Use the top 5000 words for a vocabulary.
vocabulary_size = 5000


tokenizer = tf.keras.layers.TextVectorization(
  max_tokens=vocabulary_size,
  standardize=standardize,
  output_sequence_length=max_length)
# Learn the vocabulary from the caption data.
tokenizer.adapt(caption_dataset)

optimizer = tf.keras.optimizers.Adam()          #Adam
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

# Create mappings for words to indices and indicies to words.
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

def evaluate(image):
    checkpoint_path = "ckpt/cap_multipose_v01"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

    ckpt_path = tf.train.latest_checkpoint(checkpoint_path)
    ckpt.restore(ckpt_path)


    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    max_length = 20
    attention_features_shape = 64

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)
        # if predicted_word == "<end>":
        #   return result
        if '[UNK]' in result:
          result.remove('[UNK]')
        elif '<end>' in result:
          result.remove('<end>')

        dec_input = tf.expand_dims([predicted_id], 0)
    result_final = set(result)
    result_final = list(result_final)
    # print('finalresult mdlbk:',result_final)
    # if "<end>" in result_final:
    #   result_final.remove("<end>")

    # gender = ["male", "female","men", "mens's", "women's","women","ladies","ladie's"]
    # tail_txt = open("tail_words.txt", "r")
    # tail_words = tail_txt.read()
    # tail_words = tail_words.split("\n")

    # stop_txt = open("stop_words.txt", "r")
    # stop_words = stop_txt.read()
    # stop_words = stop_words.split("\n")
    # # tail_words = ["with","end","no"]
    # # stop_words = ["xsxl","xl","xxl","sxxl","xlarge","with","end","no"]
    # for i in result_final:
    #   if i in gender:
    #       result_final.remove(i)
    #       result_final.insert(0,i)
    # for j in tail_words:
    #   if result_final[-1] == j:
    #     result_final.remove(j)
    # for k in stop_words:
    #   if k in result_final:
    #     result_final.remove(k)



    #attention_plot = attention_plot[:len(result), :]
    resultout = [x for x in result_final if len(x) > 1]
    resultout = " ".join(resultout)
    #resultout = ''.join([i for i in resultout if not i.isdigit()])
    #resultout = re.sub('\W+',' ', resultout )
    return (resultout)

def generate_filename(url):
    ext = url.split('/')[-1].split(".")[-1]
    name = generate(size=12)
    random_filename  = "{}.{}".format(name,ext)
    return random_filename


def get_image(url,path):
    # if 'bluehour' and 'secured' and 'amazonaws' in url:
    #   pass
    # #secured download code has to be written
    if 'bluehour' and 'amazonaws' in url:
       try:
           urllib.request.urlretrieve(url, path)
           return True
       except:
          return False        
    else:        
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except:
            return False
        
def clear_image(path):
    status = os.remove(path)
    return status