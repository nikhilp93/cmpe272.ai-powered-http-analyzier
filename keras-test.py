import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick
# define the text 
text = 'An example for keras hashing trick function'
# tokenizing the text 
tokens = text_to_word_sequence(text)
length = len(tokens)
final_text = hashing_trick(text, length)
print(final_text)
