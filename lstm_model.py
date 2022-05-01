from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional ,BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

from skimage.feature import hog
import pandas as pd
import numpy as np
from skimage import io,transform
import tensorflow as tf
from sklearn.metrics import  accuracy_score

def get_feature_vector_dataset(using='FLAIR', maxlen=None):
    data1 = io.imread('images/1.jpg')
    data2 = io.imread('images/2.jpg')
    data3 = io.imread('images/3.jpg')
    data4 = io.imread('images/4.jpg')
    data1 = transform.resize(data1, (64, 64))
    data2 = transform.resize(data2, (64, 64))
    data3 = transform.resize(data3, (64, 64))
    data4 = transform.resize(data4, (64, 64))
    datalist = [data1,data2,data3,data4]
    labels = [1,0,1,0]
    if maxlen:
        labels = labels[:maxlen]

    def data_generator():
        data_size = len(labels)
        random_idx = list(range(data_size))

        for i in range(data_size):
            j = random_idx[i]
            data = datalist[j]

            features = []
            for i in range(0, 3):
                features.append(hog(data[:,:,i], orientations=12, pixels_per_cell=(9, 9),
                                    cells_per_block=(1, 1), visualize=False, multichannel=False, feature_vector=True))
            features = np.array(features)
            features = features.reshape(3, -1).astype(float)
            yield features, labels[j]

    train_dataset = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int64),output_shapes=((3,588),tuple()))

    print("successfully load dataset")

    return train_dataset


train_dataset = get_feature_vector_dataset()
print(train_dataset.take(1))
for vec, l in train_dataset:
    print(vec.shape, l)


class SequenceModel:
    def __init__(self,cell_name,cell_shapes,train_dataset,val_dataset,input_shape=(3,588)):
        self.cell_name = cell_name
        self.cell_shapes = cell_shapes
        self.input_shape = input_shape
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.build_model()
        self.early_stop = EarlyStopping(patience=2)
    def get_layer(self, input_shape, state_size, return_seq = True,activation = 'relu'):
        if(self.cell_name == 'uni_lstm'):
            return LSTM(state_size, input_shape=input_shape, activation=activation, return_sequences=return_seq)
        elif(self.cell_name == 'bi_lstm'):
            return Bidirectional(LSTM(state_size, activation=activation, return_sequences=return_seq),input_shape=input_shape)
        elif(self.cell_name == 'gru'):
            return GRU(state_size, input_shape=input_shape, activation=activation, return_sequences=return_seq)

    def build_model(self):
        self.model = Sequential()
        self.model.add(self.get_layer(input_shape=self.input_shape,state_size=self.cell_shapes[0]))
        self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.2))
        for i in range(len(self.cell_shapes)-1):
            self.model.add(self.get_layer(input_shape=tuple(), state_size=self.cell_shapes[i+1]))
            self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.2))
        self.model.add(Dense(1,activation='sigmoid'))

    def train_model(self, epochs):
        print((self.model.summary()))
        self.model.compile(optimizer=Adam(learning_rate=0.01),
                      loss=binary_crossentropy,
                      metrics=['accuracy'])

        self.model.fit(self.train_dataset,epochs=epochs,validation_data=self.val_dataset,
                       validation_steps=1, callbacks=[self.early_stop])
        self.model.save(filepath='lstm.h5')


class Keras_model_FE_RNN(keras.Model):
    """ brain-tumor-detection """
    def __init__(self):
        super(Keras_model_FE_RNN, self).__init__()
        self.cell_name = 'conv_lstm'
        self.cell_shapes = [1024, 512, 256, 256]
        self.kernal_sizes = [(3,3), (3,3), (3,3), (3,3)]
        self.rnn_layers = []
        self.BN_layers = []
        self.dnn_cell_shapes = [256,64]
        self.dnn_layers = []
        self.mode = 'Bi-directional'

        if self.cell_name == 'lstm':
            for i in range(len(self.cell_shapes)-1):
                self.rnn_layers.append(tf.keras.layers.LSTM(self.cell_shapes[i], activation=tf.nn.relu,return_sequences=True))
                self.BN_layers.append(tf.keras.layers.BatchNormalization())
                #self.dropout = tf.keras.layers.Dropout(0.5)
            self.Last_rnn_layer = tf.keras.layers.LSTM(self.cell_shapes[len(self.cell_shapes) - 1],
                                                       activation=tf.nn.relu)
        if self.cell_name == 'conv_lstm':
            for i in range(len(self.cell_shapes)-1):
                self.rnn_layers.append(tf.keras.layers.ConvLSTM2D(self.cell_shapes[i], self.kernal_sizes[i], activation=tf.nn.relu,return_sequences=True))
                self.BN_layers.append(tf.keras.layers.BatchNormalization())
                #self.dropout = tf.keras.layers.Dropout(0.5)
            self.Last_rnn_layer = tf.keras.layers.ConvLSTM2D(self.cell_shapes[len(self.cell_shapes) - 1], self.kernal_sizes[len(self.cell_shapes) - 1],
                                                       activation=tf.nn.relu)
            self.flatten_layer = tf.keras.layers.Flatten()
        if self.mode == 'Bi-directional':
            for i in range(len(self.cell_shapes)-1):
                self.rnn_layers[i] = tf.keras.layers.Bidirectional(self.rnn_layers[i])
            self.Last_rnn_layer = tf.keras.layers.Bidirectional(self.Last_rnn_layer)

        for i in range(len(self.dnn_cell_shapes)):
             self.dnn_layers.append(tf.keras.layers.Dense(self.dnn_cell_shapes[i],activation='relu'))
        self.DNN_layer = tf.keras.layers.Dense(1,activation='sigmoid')

    def call(self, inputs):
        x = inputs
        for i in range(len(self.rnn_layers)):
            x = self.rnn_layers[i](x)
            x = self.BN_layers[i](x)
        x = self.Last_rnn_layer(x)
        x = self.flatten_layer(x)
        for d in self.dnn_layers:
            x = d(x)
        output = self.DNN_layer(x)
        # if training:
        #     x = self.dropout(x, training=training)
        return output


def train(train_dataset, validation_dataset, model, batch_size=5, EPOCHS=3):
    dataset = train_dataset.batch(batch_size)
    val_dataset = validation_dataset.batch(batch_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(data, label):
        with tf.GradientTape() as tape:
            output = model(data)
            loss = loss_func(label, output)
        score = accuracy_score(label, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, score

    def validation_step(data, label):
        output = model(data)
        loss = loss_func(label, output)
        score = accuracy_score(label, output)
        return loss, score

    for epoch in range(EPOCHS):
        print(f"start epoch {epoch}")

        for itr, (data, label) in enumerate(dataset):
            loss, score = train_step(data, label)
            print(loss, score)
        print('validation')
        if epoch % 1 == 0:
            for itr, (data, label) in enumerate(val_dataset):
                loss, score = validation_step(data, label)
                print('validation', loss, score)

model = Keras_model_FE_RNN()
model.build(input_shape = (None,3,588))
print(model.summary())

# model.compile(optimizer=Adam(learning_rate=0.01),
#                       loss=binary_crossentropy,
#                       metrics=['accuracy'])
# model.fit(train_dataset.batch(2),epochs=20,validation_data=train_dataset.batch(2),
#                        validation_steps=1)
# for i,j in train_dataset.batch(1):
#     d = i
#     l = j
#     print(i,j)
# model = SequenceModel(cell_name='bi_lstm',input_shape=(3,588),cell_shapes=[512,128],
#                       train_dataset=train_dataset.batch(2),val_dataset=train_dataset.batch(2))
#model.train_model(epochs=100)






