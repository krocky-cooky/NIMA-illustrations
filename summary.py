import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import utils
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import json
from efficientnet.keras import EfficientNetB7,EfficientNetB4


import os,sys
sys.path.append(os.path.dirname(__file__))

np.random.seed(100)


def EfficientNetWithRatio(
    input1_shape,
    input2_shape,
    encode_dim,
    output_dim
):
"""
モデル
"""
    input1 = kl.Input(shape = input1_shape)
    input2 = kl.Input(shape = input2_shape)
    efficient_net = EfficientNetB4(
        weights = 'noisy-student',
        include_top = False,
        input_tensor = input1,
    )

    for layer in efficient_net.layers:
        layer.trainable = True

    bottleneck = efficient_net.output

    pool_out = kl.GlobalAveragePooling2D()(bottleneck)
    ratio_out = kl.Dense(5,activation = 'linear')(input2)
    _ = kl.concatenate([pool_out,ratio_out])##　アスペクト比をベクトルに結合
    _ = kl.Dense(encode_dim,activation = 'relu')(_)
    outputs = kl.Dense(output_dim,activation = 'softmax')(_)

    model = Model(inputs = [input1,input2],outputs = outputs)

    return model

class MultiDataGenerator(tf.keras.utils.Sequence):
    """
    画像をバッチごとに読み込むために必要なコード
    datageneratorというらしい
    """
    def __init__(
        self,
        data,
        target,
        image_path,
        batch_size
    ):
        

        self.id_ = data[:,0].astype(np.int32)
        self.aspect_ratio_ = data[:,1]
        self.t_ = target
        self.batch_size = batch_size
        self.image_path = image_path
        self.data_size = data.shape[0]
        self.n_batches = (self.data_size-1) // self.batch_size + 1

    def __getitem__(self,idx):
        start = self.batch_size*idx
        end = min(start + self.batch_size,self.data_size)
        x_batch = self.id_[start:end]
        t_batch = self.t_[start:end]
        aspect_ratio_batch = self.aspect_ratio_[start:end]
        img_batch = list()

        for id in x_batch:
            image = np.load(self.image_path + '/' + str(id) + '.npy')
            img_batch.append(image/255)
        img_batch = np.array(img_batch)
        return [img_batch,aspect_ratio_batch],t_batch

    def __len__(self):
        return self.n_batches

    def on_epoch_end(self):
        self.id_,self.aspect_ratio_,self.t_ = utils.shuffle(self.id_,self.aspect_ratio_,self.t_)


def EMD(t,preds):
    """
    Earth Mover's Distance
    """
    return tf.reduce_mean(tf.reduce_sum(tf.math.cumsum(preds-t,axis = 1)**2,axis = 1))

class TrainerV4(object):
    def __init__(
        self,
        input1_shape,
        input2_shape,
        encode_dim,
        output_dim,
        model = 'efficient_net',
        loss = 'emd',
        learning_rate = 0.1
    ):
        self.model = None
        if model == 'efficient_net':
            self.model = EfficientNetWithRatio(input1_shape,input2_shape,encode_dim,output_dim)
        #elif model == 'wide_res_net':
        #    self.model = WideResNetWithRatio(input_shape,output_dim)
        else:
            raise Exception('no match model name')
        optimizer = tf.keras.optimizers.SGD(
            learning_rate = learning_rate,
            momentum = 0.1
        )
        loss_func = None
        if loss == 'emd':
            loss_func = EMD
        elif loss == 'categorical_crossentropy':
            loss_func = 'categorical_crossentropy'
        else:
            raise Exception('no match loss function')
        self.model.compile(
            optimizer = optimizer,
            loss = loss_func,
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
        image_path,
        save_name
    ):
        train_gen = MultiDataGenerator(
            x_train,
            t_train,
            image_path = image_path,
            batch_size = batch_size
        )
        val_gen = MultiDataGenerator(
            x_val,
            t_val,
            image_path = image_path,
            batch_size = batch_size
        )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                save_name,
                monitor = 'val_loss',
                verbose = 1,
                save_best_only = True,
                mode = 'min'
                )
        ]
        
        
        self.history = self.model.fit_generator(
            train_gen,
            len(train_gen),
            epochs = 30,
            validation_data = val_gen,
            validation_steps = len(val_gen),
            callbacks = callbacks,
        )

    def evaluate(
        self,
        x_test,
        t_test,
        batch_size,
        image_path,
    ):
        test_gen = MultiDataGenerator(
            x_test,
            t_test,
            image_path = image_path,
            batch_size = batch_size
        )

        preds = self.model.predict_generator(
            test_gen,
            len(test_gen),
        )
        acc = accuracy_score(np.argmax(t_test,axis = 1),np.argmax(preds,axis = 1))
        cm = confusion_matrix(np.argmax(t_test,axis = 1),np.argmax(preds,axis = 1))
        print(acc,cm)
        return (acc,cm)

    def predict(
        self,
        ids,
        batch_size,
        image_path
    ):
        test_gen = MultiDataGenerator(
            ids,
            ids,
            image_path = image_path,
            batch_size = batch_size
        )
        preds = self.model.predict_generator(
            test_gen,
            len(test_gen)
        )
        print(preds)
        return preds

if __name__ == '__main__':

    df = pd.read_csv('./target_available_v2.csv')

    datas = list()
    targets = list()
    tmp = df[df['label'] == 0]
    idx = np.random.choice([_ for _ in range(tmp.shape[0])],size = 22000,replace = False)
    datas.append(tmp[['illust_id','aspect_ratio_norm']].iloc[idx])
    targets.append([0 for _ in range(22000)])

    tmp = df[df['label'] == 1]
    idx = np.random.choice([_ for _ in range(tmp.shape[0])],size = 22000,replace = False)
    datas.append(tmp[['illust_id','aspect_ratio_norm']].iloc[idx])
    targets.append([1 for _ in range(22000)])

    tmp = df[df['label'] == 2]
    idx = np.random.choice([_ for _ in range(tmp.shape[0])],size = 22000,replace = False)
    datas.append(tmp[['illust_id','aspect_ratio_norm']].iloc[idx])
    targets.append([2 for _ in range(22000)])

    tmp = df[df['label'] == 3]
    datas.append(tmp[['illust_id','aspect_ratio_norm']].to_numpy())
    targets.append(tmp['label'].to_numpy())

    tmp = df[df['label'] == 4]
    datas.append(tmp[['illust_id','aspect_ratio_norm']].to_numpy())
    targets.append(tmp['label'].to_numpy())

    datas = np.vstack(datas)
    targets = np.hstack(targets)
    targets = np.identity(5)[targets]


    x_train,x_test,t_train,t_test = train_test_split(datas,targets,shuffle = True,test_size = 0.1,random_state = 0)
    x_train,x_val,t_train,t_val = train_test_split(x_train,t_train,shuffle = True,test_size = 0.1,random_state = 0)


    trainer = TrainerV4(
        input1_shape = (128,128,3),
        input2_shape = (1,),
        encode_dim = 1000
        output_dim = 5,
        model = 'efficient_net',
    )

    trainer.train(
        x_train,
        t_train,
        x_val,
        t_val,
        epochs = 25,
        batch_size = 512,
        image_path = 'images_128_128_npy_v2'
    )
