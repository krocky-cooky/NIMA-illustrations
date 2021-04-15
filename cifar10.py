import os,sys
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from efficientnet.keras import EfficientNetB4


np.random.seed(100)

def Network(
    input_shape,
    encode_dim,
    output_dim
):
    input_layer = kl.Input(shape = input_shape)
    efficient_net = EfficientNetB4(
        weights = 'noisy-student',
        include_top = False,
        input_tensor = input_layer,
    )

    for layer in efficient_net.layers:
        layer.trainable = True

    bottleneck = efficient_net.output
    _ = kl.GlobalAveragePooling2D()(bottleneck)
    _ = kl.Dense(encode_dim,activation = 'relu')(_)
    outputs = kl.Dense(output_dim,activation = 'softmax')(_)
    model = Model(inputs = input_layer,outputs = outputs)

    return model

class Trainer(object):
    def __init__(
        self,
        input_shape,
        encode_dim,
        output_dim,
        learning_rate = 0.1
    ):
        self.model = Network(
            input_shape = input_shape,
            encode_dim = encode_dim,
            output_dim = output_dim
            )
        optimizer = tf.keras.optimizers.SGD(
            learning_rate = learning_rate,
            momentum = 0.1
        )
        self.model.compile(
            optimizer = optimizer,
            loss = 'categorical_crossentropy',
            metrics = ['acc']
        )

    def train(
        self,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs,
        batch_size,
        save_name
    ):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                save_name,
                monitor = 'val_loss',
                verbose = 1,
                save_best_only = True,
                mode = 'min'
            )
        ]
        self.history = self.model.fit(
            x_train,
            t_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (x_val,t_val),
            callbacks = callbacks
        )

    def evaluate(
        self,
        x_test,
        t_test,
        batch_size,
    ):
        preds = self.model.predict(
            x_train,
            batch_size - batch_size,
        )
        acc = accuracy_score(np.argmax(t_test,axis = 1),np.argmax(preds,axis = 1))
        cm = confusion_matrix(np.argmax(t_test,axis = 1),np.argmax(preds,axis = 1))
        print('accuracy: {}\nconfusion_matrix:\n{}'.format(acc,cm))
        return (acc,cm)

    


if __name__ == '__main__':
    (x_train,t_train),(x_test,t_test) = cifar10.load_data()
    x_train = x_train/255
    x_test = x_test/255
    t_train = np.identity(10)[t_train.reshape(-1)]
    t_test = np.identity(10)[t_test.reshape(-1)]

    x_train,x_val,t_train,t_val = train_test_split(x_train,t_train,test_size = 0.2,random_state = 0)

    trainer = Trainer(
        input_shape = (32,32,3),
        encode_dim = 500,
        output_dim = 10,
        learning_rate = 0.1,
    )

    trainer.train(
        x_train,
        t_train,
        x_val,
        t_val,
        epochs = 30,
        batch_size = 512,
        save_name = 'cifar10_efnet'
    )

    trainer.evaluate(
        x_test,
        t_test,
        batch_size = 512,
    )


    

