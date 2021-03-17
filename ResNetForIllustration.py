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
import json

import os,sys
sys.path.append(os.path.dirname(__file__))

from models import ResNet,WideResNet,EfficientNet,EMD

class DataLoader(object):
    def __init__(self,file):
        df = pd.read_csv(file)
        data_train,data_test,_,_ = train_test_split(df,df['label'],stratify = df['label'],test_size = 0.2)
        self.x_test = data_test['illust_id'].to_numpy()
        self.t_test = data_test['label'].to_numpy()
        self.t_test = np.identity(5)[self.t_test]
        self.data_train,data_val,_,_ = train_test_split(data_train,data_train['label'],stratify = data_train['label'],test_size = 0.1)
        self.x_val = data_val['illust_id'].to_numpy()
        self.t_val = data_val['label'].to_numpy()
        self.t_val = np.identity(5)[self.t_val]
        
        self.val_size = self.x_val.shape[0]
        self.test_size = self.x_test.shape[0]
        
            
    def get_train_data(self):
        datas = list()
        targets = list()
        tmp = self.data_train[self.data_train['label'] == 0]
        d = np.random.choice(tmp['illust_id'].to_numpy(),size = 20000,replace = False)
        datas.append(d)
        targets.append([0 for _ in range(d.shape[0])])
        
        tmp = self.data_train[self.data_train['label'] == 1]
        d = np.random.choice(tmp['illust_id'].to_numpy(),size = 20000,replace = False)
        datas.append(d)
        targets.append([1 for _ in range(d.shape[0])])
        
        tmp = self.data_train[self.data_train['label'] == 2]
        d = np.random.choice(tmp['illust_id'].to_numpy(),size = 20000,replace = False)
        datas.append(d)
        targets.append([2 for _ in range(d.shape[0])])
        
        tmp = self.data_train[self.data_train['label'] == 3]
        datas.append(tmp['illust_id'].to_numpy())
        targets.append(tmp['label'].to_numpy())
        
        tmp = self.data_train[self.data_train['label'] == 4]
        datas.append(tmp['illust_id'].to_numpy())
        targets.append(tmp['label'].to_numpy())
        
        datas = np.hstack(datas)
        targets = np.hstack(targets)
        targets = np.identity(5)[targets]
        return utils.shuffle(datas,targets)


class MnistTrainer(object):
    def __init__(
        self,
        input_shape,
        output_dim,
        patience = 4,
        structure = 'wide_res_net',
    ):
        self.model = None
        if structure == 'wide_res_net':
            self.model = WideResNet(
                input_shape = input_shape,
                output_dim = output_dim
            )
        elif structure == 'res_net':
            self.model = ResNet(
                input_shape = input_shape,
                output_dim = output_dim
            )
        else:
            raise Exception('no structure')
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
        self.train_loss = tf.keras.metrics.Mean()
        self.train_acc = tf.keras.metrics.CategoricalAccuracy()
        self.val_loss = tf.keras.metrics.Mean()
        self.val_acc = tf.keras.metrics.CategoricalAccuracy()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.es = {
            'loss': float('inf'),
            'patience': patience,
            'step': 0
        }
        self.save_dir = './logs'
        if not os.path.exists(self.save_dir):
            os.mkdir('logs')

    def train(
        self,
        data_loader,
        epochs,
        batch_size,
        early_stopping = False
    ):
        
        
        n_batches_val = data_loader.val_size // batch_size
        x_val = data_loader.x_val
        t_val = data_loader.t_val

        for epoch in range(epochs):
            x_,t_ = data_loader.get_train_data()
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            self.train_acc.reset_states()
            self.val_acc.reset_states()
            
            for batch in tqdm(range(n_batches_train)):
                start = batch * batch_size
                end = start + batch_size
                x_batch = x_[start:end]
                t_batch = t_[start:end]
                self.train_step(x_batch,t_batch)

            for batch in tqdm(range(n_batches_val)):
                start = batch * batch_size
                end = start + batch_size
                x_batch = x_val[start:end]
                t_batch = t_val[start:end]
                self.val_step(x_batch,t_batch)
            
            self.history['train_loss'].append(self.train_loss.result())
            self.history['val_loss'].append(self.val_loss.result())
            self.history['train_acc'].append(self.train_acc.result())
            self.history['val_acc'].append(self.val_acc.result())
            print('epoch {} => train_loss: {},  train_acc: {}, val_loss: {}, val_acc: {}'.format(
                epoch + 1,
                self.train_loss.result(),
                self.train_acc.result(),
                self.val_loss.result(),
                self.val_acc.result()
            ))

            if early_stopping:
                if self.early_stopping(self.val_loss.result()):
                    break

        fig = plt.figure(figsize = (20,20))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        ax1.plot(self.history['train_loss'])
        ax2.plot(self.history['train_acc'])
        ax3.plot(self.history['val_loss'])
        ax4.plot(self.history['val_acc'])
        ax1.set_title('train_loss')
        ax2.set_title('train_acc')
        ax3.set_title('val_loss')
        ax4.set_title('val_acc')
        plt.show()

    def train_step(self,x,t):
        with tf.GradientTape() as tape:
            preds = self.model(x)
            loss = self.criterion(t,preds)
        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        self.train_loss(loss)
        self.train_acc(t,preds)

    def val_step(self,x,t):
        preds = self.model(x)
        loss = self.criterion(t,preds)
        self.val_loss(loss)
        self.val_acc(t,preds)

    def evaluate(self,x_test,t_test):
        accuracy = tf.metrics.CategoricalAccuracy()
        preds = self.model(x_test)
        loss = self.criterion(preds,t_test)
        accuracy(t_test,preds)
        print('accuracy: {}, loss: {}'.format(
            accuracy.result(),
            loss
        ))
        return (accuracy.result().numpy(),loss.numpy())

    def save(self,name):
        path = self.save_dir +'/' + name
        self.model.save_weights(path)
    
    def load(self,name):
        path = self.save_dir +'/' + name
        self.model.load_weights(path)

    def early_stopping(self,loss):
        if loss > self.es['loss']:
            self.es['step'] += 1
            if self.es['step'] > self.es['patience']:
                print('early stopping')
                self.load('early_stopping_saving')
                return True
        else:
            self.es['loss'] = loss
            self.es['step'] = 0
            self.save('early_stopping_saving')

        return False

    

class Trainer(object):
    def __init__(
        self,
        input_shape = (128,128,3),
        output_dim = 10,
        patience = 5,
        structure = 'wide_res_net',
        loss = 'categorical_crossentropy',
        name = 'latest'
        ):
        self.model = None
        if structure == 'wide_res_net':
            self.model = WideResNet(
                input_shape = input_shape,
                output_dim = output_dim
            )
        elif structure == 'res_net':
            self.model = ResNet(
                input_shape = input_shape,
                output_dim = output_dim
            )
        else:
            raise Exception('no structure')
        if loss == 'categorical_crossentropy':
            self.criterion = tf.keras.losses.CategoricalCrossentropy() 
        elif loss == 'emd':
            self.criterion = EMD
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.1,
            momentum = 0.1
            )
        self.train_acc = tf.keras.metrics.CategoricalAccuracy()
        self.train_loss = tf.keras.metrics.Mean()
        self.val_acc = tf.keras.metrics.CategoricalAccuracy()
        self.val_loss = tf.keras.metrics.Mean()
        self.history = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
            'start': 0,
        }
        self.es = {
            'loss': float('inf'),
            'patience': patience,
            'step': 0
        }
        self.save_dir = './logs'
        if not os.path.exists(self.save_dir):
            os.mkdir('logs')
        self.name = name
    

    def train(
        self,
        data_loader,
        epochs,
        batch_size,
        image_path,
        early_stopping = False
    ):
        n_batches_val = data_loader.val_size // batch_size
        x_val = data_loader.x_val
        t_val = data_loader.t_val
        start = self.history['start']


        for epoch in range(start,epochs):
            self.history['start'] = epoch
            x_,t_ = data_loader.get_train_data()
            n_batches_train = x_.shape[0] // batch_size
            self.train_acc.reset_states()
            self.train_loss.reset_states()
            self.val_acc.reset_states()
            self.val_loss.reset_states()
            
            for batch in tqdm(range(n_batches_train)):
                start = batch * batch_size
                end = start + batch_size
                x_batch = x_[start:end]
                t_batch = t_[start:end]
                img_batch = self.get_image(image_path,x_batch)
                self.train_step(img_batch,t_batch)


            for batch in tqdm(range(n_batches_val + 1)):
                start = batch * batch_size
                end = start + batch_size
                end = min(end,x_val.shape[0])
                x_batch = x_val[start:end]
                t_batch = t_val[start:end]
                img_batch = self.get_image(image_path,x_batch)
                self.val_step(img_batch,t_batch)
            
            self.history['train_loss'].append(self.train_loss.result().numpy())
            self.history['val_loss'].append(self.val_loss.result().numpy())
            self.history['train_acc'].append(self.train_acc.result().numpy())
            self.history['val_acc'].append(self.val_acc.result().numpy())
            print('epoch {} => train_loss: {},  train_acc: {}, val_loss: {}, val_acc: {}'.format(
                epoch + 1,
                self.train_loss.result(),
                self.train_acc.result(),
                self.val_loss.result(),
                self.val_acc.result()
            ))


            if self.early_stopping(self.val_loss.result()):
                if early_stopping:
                    break
            self.save(self.name)

        self.plot()




    def get_image(
        self,
        image_path,
        x_batch,
        standard = 255
    ):
        img_batch = list()
        for id in x_batch:
            image = np.load(image_path + '/' + str(id) + '.npy')
            img_batch.append(image/standard)
            
        return np.array(img_batch)

    def train_step(self,x,t):
        with tf.GradientTape() as tape:
            preds = self.model(x)
            loss = self.criterion(t,preds)
        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        self.train_loss(loss)
        self.train_acc(t,preds)

    def val_step(self,x,t):
        preds = self.model(x)
        loss = self.criterion(t,preds)
        self.val_loss(loss)
        self.val_acc(t,preds)

    def evaluate(self,x_test,t_test,batch_size = 1000):
        loss = tf.keras.metrics.Mean()
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        n_batches = x_test.shape[0] // batch_size
        for batch in tqdm(range(n_batches)):
            start = batch_size * batch
            end = start + batch_size
            x_batch = x_test[start:end]
            t_batch = t_test[start:end]
            img_batch = self.get_image(x_batch)
            preds = self.model(img_batch)
            loss(self.criterion(t_batch,preds))
            accuracy(t_batch,preds)

        start = batch_size * n_batches
        end = x_test.shape[0]
        x_batch = x_test[start:end]
        t_batch = t_test[start:end]
        img_batch = self.get_image(x_batch)
        preds = self.model(img_batch)
        loss(self.criterion(t_batch,preds))
        accuracy(t_batch,preds)
        print('loss: {}, accuracy: {}'.format(
            loss.result(),
            accuracy.result()
        )
        )
        return (loss.result().numpy(),accuracy.result().numpy())

    def early_stopping(self,loss):
        name = self.name + '_es'
        if loss > self.es['loss']:
            self.es['step'] += 1
            if self.es['step'] > self.es['patience']:
                print('early stopping')
                return True
        else:
            self.es['loss'] = loss
            self.es['step'] = 0
            self.save(name)

        return False

    def save(self,name):
        path = self.save_dir +'/' + name
        self.model.save_weights(path)
        dic = {
            'history': self.history,
            'es': self.es
        }
        with open(self.save_dir + '/' + name + '.json','w') as f:
            json.dump(dic,f)
        
    def load(self,name):
        path = self.save_dir +'/' + name
        self.model.load_weights(path)
        with open(self.save_dir + '/' + name + '.json','r') as f:
            data = f.read()
        dic = json.load(data)
        self.hisotory = dic['history']
        self.es = dic['es']

    def plot(self):
        fig = plt.figure(figsize = (10,10))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        ax1.plot(self.history['train_loss'])
        ax2.plot(self.history['train_acc'])
        ax3.plot(self.history['val_loss'])
        ax4.plot(self.history['val_acc'])
        ax1.set_title('train_loss')
        ax2.set_title('train_acc')
        ax3.set_title('val_loss')
        ax4.set_title('val_acc')
        plt.show()
    
class TrainerV2(object):
    def __init__(
        self,
        input_shape,
        output_dim,
        model = 'efficient_net'
    ):
        self.model = None
        if model == 'efficient_net':
            self.model = EfficientNet(input_shape,output_dim)
        elif model == 'wide_res_net':
            self.model = WideResNet(input_shape,output_dim)
        else:
            raise Exception('no match model name')
        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.1,
            momentum = 0.1
        )
        self.model.compile(
            optimizer = optimizer,
            loss = EMD,
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
        image_path
    ):
        train_gen = DataGenerator(
            x_train,
            t_train,
            image_path = image_path,
            batch_size = batch_size
        )
        val_gen = DataGenerator(
            x_val,
            t_val,
            image_path = image_path,
            batch_size = batch_size
        )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                './logs',
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
        test_gen = DataGenerator(
            x_test,
            t_test,
            image_path = image_path,
            batch_size = batch_size
        )

        self.model.evaluate_generator(
            test_gen,
            len(test_gen),
        )

class TrainerV3(object):
    def __init__(self,input_shape,output_dim,model = 'efficient_net'):
        self.model = None
        if model == 'efficient_net':
            self.model = EfficientNetWithMultiOutput(input_shape,output_dim)
        elif model == 'wide_res_net':
            self.model = WideResNetWithMultiOutput(input_shape,output_dim)
        else:
            raise Exception('no match model name')
        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.1,
            momentum = 0.1
        )
        self.model.compile(
            optimizer = optimizer,
            loss = {
                'bookmark': EMD,
                'aspect_ratio': 'mse',
            },
            loss_weights = {
                'bookmark': 5.,
                'aspect_ratio': 1.
            },
            metrics = {
                'bookmark': 'accuracy',
                'aspect_ratio': 'mae'
            }
        )
    
    def train(
        self,
        x_train,
        aspect_train,
        t_train,
        x_val,
        aspect_val,
        t_val,
        epochs,
        batch_size,
        image_path,
        save_name = 'best'
    ):
        train_gen = MultiDataGenerator(
            x_train,
            aspect_train,
            t_train,
            image_path = image_path,
            batch_size = batch_size
        )
        val_gen = MultiDataGenerator(
            x_val,
            aspect_val,
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
            epochs = epochs,
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
            aspect_test,
            t_test,
            image_path = image_path,
            batch_size = batch_size
        )

        self.model.evaluate_generator(
            test_gen,
            len(test_gen),
        )


class EfficientNetMnistTrainer(object):
    def __init__(self,input_shape,output_dim):
        self.model = EfficientNet(input_shape,output_dim)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.1,
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
    ):
        self.history = self.model.fit(
            x_train,
            t_train,
            batch_size = batch_size,
            epochs = 30,
            validation_data = (x_val,t_val)
        )
        
    
    

        
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data,
        target,
        image_path,
        batch_size = 512
    ):
        

        self.x_ = data
        self.t_ = target
        self.batch_size = batch_size
        self.image_path = image_path
        self.data_size = data.shape[0]
        self.n_batches = (self.data_size-1) // self.batch_size + 1

    def __getitem__(self,idx):
        start = self.batch_size*idx
        end = min(start + self.batch_size,self.data_size)
        x_batch = self.x_[start:end]
        t_batch = self.t_[start:end]
        img_batch = list()

        for id in x_batch:
            image = np.load(self.image_path + '/' + str(id) + '.npy')
            img_batch.append(image/255)
        img_batch = np.array(img_batch)
        return img_batch,t_batch

    def __len__(self):
        return self.n_batches

    def on_epoch_end(self):
        self.x_,self.t_ = utils.shuffle(self.x_,self.t_)

class MultiDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        id_data,
        aspect_ratio_data,
        target,
        image_path,
        batch_size = 512
    ):
        

        self.id_ = id_data
        self.aspect_ratio_ = aspect_ratio_data
        self.t_ = target
        self.batch_size = batch_size
        self.image_path = image_path
        self.data_size = id_data.shape[0]
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
        return img_batch,[t_batch,aspect_ratio_batch]

    def __len__(self):
        return self.n_batches

    def on_epoch_end(self):
        self.x_,self.t_ = utils.shuffle(self.x_,self.t_)







