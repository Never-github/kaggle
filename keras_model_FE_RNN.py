from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional ,BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from skimage.feature import hog
import pandas as pd
import numpy as np
from skimage import io,transform
import tensorflow as tf
from sklearn.metrics import accuracy_score
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

class Keras_model_FE_RNN(keras.Model):
    """ brain-tumor-detection """
    def __init__(self):
        super(Keras_model_FE_RNN, self).__init__()
        self.cell_name = 'lstm'
        self.cell_shapes = [128,128,128]
        self.rnn_layers = []
        self.BN_layers = []
        self.mode = 'Bi-directional'

        if self.cell_name == 'lstm':
            for i in range(len(self.cell_shapes)-1):
                self.rnn_layers.append(tf.keras.layers.LSTM(self.cell_shapes[i], activation=tf.nn.relu,return_sequences=True))
                self.BN_layers.append(tf.keras.layers.BatchNormalization())
                #self.dropout = tf.keras.layers.Dropout(0.5)
            self.Last_rnn_layer = tf.keras.layers.LSTM(self.cell_shapes[len(self.cell_shapes) - 1],
                                                       activation=tf.nn.relu)

        if self.mode == 'Bi-directional':
            for i in range(len(self.cell_shapes)-1):
                self.rnn_layers[i] = tf.keras.layers.Bidirectional(self.rnn_layers[i])
            self.Last_rnn_layer = tf.keras.layers.Bidirectional(self.Last_rnn_layer)

        self.DNN_layer = tf.keras.layers.Dense(1,activation='sigmoid')
    def call(self, inputs):
        x = inputs
        for i in range(len(self.rnn_layers)):
            x = self.rnn_layers[i](x)
            x = self.BN_layers[i](x)
        x = self.Last_rnn_layer(x)
        output = self.DNN_layer(x)
        # if training:
        #     x = self.dropout(x, training=training)
        return output


def train(train_dataset, validation_dataset, model, batch_size=5, EPOCHS=3, submission=False):
    dataset = train_dataset.batch(batch_size).repeat(EPOCHS)
    val_dataset = validation_dataset.batch(batch_size).repeat(EPOCHS)
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.BinaryCrossentropy()

    #@tf.function
    def train_step(data, label):
        with tf.GradientTape() as tape:
            output = model(data)
            loss = loss_func(label, output)
        output = [1 if i>0.5 else 0 for i in output]
        score = accuracy_score(label,output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, score

    def validation_step(data,label):
        output = model(data)
        loss = loss_func(label, output)
        output = [1 if i > 0.5 else 0 for i in output]
        score = accuracy_score(label, output)
        return loss, score

    for epoch in range(EPOCHS):
        print(f"start epoch {epoch+1}")
        print('training...')
        avg_loss = 0
        avg_score = 0
        count = 0
        train_bar = tqdm(enumerate(dataset))
        for itr, (data, label) in train_bar:
            loss, score = train_step(data, label)
            avg_loss += loss
            avg_score += score
            count += 1
            train_bar.set_postfix(train_loss=avg_loss/count, train_acc = avg_score/count)
        print(f'train_loss={avg_loss / count},train_acc={avg_score / count}')
        print('validation...')
        if epoch % 1 == 0:
            avg_loss = 0
            avg_score = 0
            count = 0
            val_bar = tqdm(val_dataset)
            for itr, (data, label) in val_bar:
                loss, score = validation_step(data,label)
                avg_loss += loss
                avg_score += score
                count += 1
                val_bar.set_postfix(val_loss=avg_loss/count,val_acc=avg_score/count)
            print(f'val_loss={avg_loss/count},val_acc={avg_score/count}')
        #model.save_weights(f'dnn_{avg_loss / count}.h5')




model = Keras_model_FE_RNN()
model.build(input_shape = (None,3,588))
model.summary()
train(train_dataset,train_dataset,model,2,5)


