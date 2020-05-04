"""
Models to be used by the trainer
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, Layer
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers

def simple_mlp(input_shape):

    input = Input(shape=(input_shape,))
    x = Dense(2, activation='relu')(input)
    output = Dense(1, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model

def conv_mlp(input_shape):

    """
    Model with multiple conv layer/maxpool layer blocks -> mlp
    """

    NUM_FILTERS = 512
    input = Input(shape=(input_shape,))
    """
    x = Conv1D(filters=NUM_FILTERS, kernel_size=2, padding="same", kernel_regularizer=regularizers.l1(0.01))(input)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(filters=NUM_FILTERS*2, kernel_size=2, padding="same", kernel_regularizer=regularizers.l1(0.01))(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(filters=NUM_FILTERS*4, kernel_size=2, padding="same", kernel_regularizer=regularizers.l1(0.01))(x)
    x = tf.reduce_max(x, axis=1)
    x = Flatten()(x)
    """

    x = Dense(4096, activation='relu', use_bias=False)(input)
    x = Dense(2048, activation='relu', use_bias=False)(x)
    x = Dense(1024, activation='relu', use_bias=False)(x)
    x = Dense(512, activation='relu', use_bias=False)(x)
    x = Dense(256, activation='relu', use_bias=False)(x)
    x = Dense(128, activation='relu', use_bias=False)(x)
    x = Dense(64, activation='relu', use_bias=False)(x)
    x = Dense(32, activation='relu', use_bias=False)(x)

    output = Dense(1, activation='softmax', use_bias=False)(x)

    model = Model(inputs=input, outputs=output)

    return model

"""
example of two models from another project


def simple_model(num_occupation, num_gender, embedding_dim, vocab_size, filters):
    
    Model of a very simple multi-output keras model.
    Consists of: Input -> Embedding -> Conv1D -> Flatten |---> Occupation output
                                                         |---> Gender output
    
    main_input = Input(shape=(None,))  # Input is a variable length list of subwords.
    # (So input_shape should be None).

    #input_dim should be the vocab of sentencepiece model,
    #output_dim is size of embeddings
    x = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=(None,))(main_input)
    x = Conv1D(filters=filters, kernel_size=2, padding="same", kernel_regularizer=regularizers.l1(0.01))(x)
    x = tf.reduce_max(x, axis=1)
    x = Flatten()(x)

    occupation_output = Dense(num_occupation, activation='sigmoid', name="occupation",
                              use_bias=False, kernel_regularizer=regularizers.l1(0.01))(x)
    gender_output = Dense(num_gender, activation='softmax', name="gender",
                          use_bias=False, kernel_regularizer=regularizers.l1(0.01))(x)

    model = Model(inputs=main_input, outputs=[occupation_output, gender_output])

    return model

def conv_mlp(num_occupation, num_gender, embedding_dim, vocab_size, filters):
    
    Model with conv layer -> mlp
    
    main_input = Input(shape=(None,), name='input')  # Input is a variable length list of subwords.
    # (So input_shape should be None).

    #input_dim should be the vocab of sentencepiece model,
    #output_dim is size of embeddings
    x = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=(None,), name='embedding')(main_input)
    x = Conv1D(filters=filters, kernel_size=2, padding="same", kernel_regularizer=regularizers.l1(0.01), name='conv1')(x)
    x = tf.reduce_max(x, axis=1)
    x = Flatten()(x)

    x = Dense(4096, activation='relu', use_bias=False, name='dense1')(x)
    x = Dense(2048, activation='relu', use_bias=False, name='dense2')(x)
    x = Dense(1024, activation='relu', use_bias=False, name='dense3')(x)
    x = Dense(512, activation='relu', use_bias=False, name='target')(x)
    occupation_input = Dense(1024, activation='relu', use_bias=False)(x)
    x = Dense(256, activation='relu', use_bias=False)(x)
    x = Dense(128, activation='relu', use_bias=False)(x)
    x = Dense(64, activation='relu', use_bias=False)(x)
    x = Dense(32, activation='relu', use_bias=False)(x)
    gender_input = Dense(16, activation='relu', use_bias=False)(x)

    occupation_output = Dense(num_occupation, activation='sigmoid', name="occupation",
                              use_bias=False, kernel_regularizer=regularizers.l1(0.01))(occupation_input)
    gender_output = Dense(num_gender, activation='softmax', name="gender",
                          use_bias=False, kernel_regularizer=regularizers.l1(0.01))(gender_input)

    model = Model(inputs=main_input, outputs=[occupation_output, gender_output])

    return model

"""


