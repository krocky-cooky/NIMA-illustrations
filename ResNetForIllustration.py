import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm


class ResBlock(Model):
    def __init__(
        self,
        input_channels,
        output_channels,
    ):
        super().__init__()
        bneck_channels = output_channels // 4
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(
            bneck_channels,
            kernel_size = 1,
            strides = 1,
            padding = 'valid',
            use_bias = False
        )
        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(
            bneck_channels,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            use_bias = False
        )
        self.bn3 = kl.BatchNormalization()
        self.av3 = kl.Activation(tf.nn.relu)
        self.conv3 = kl.Conv2D(
            output_channels,
            kernel_size = 1,
            strides = 1,
            padding = 'valid',
            use_bias = False
        )

        self.shortcut = self._scblock(input_channels,output_channels)
        self.add = kl.Add()

       

    def _scblock(
        self,
        input_channels,
        output_channels
    ):
        if input_channels == output_channels:
            return lambda x : x
        else:
            self.bn_sc = kl.BatchNormalization()
            self.conv_sc = kl.Conv2D(
                output_channels,
                kernel_size = 1,
                strides = 1,
                padding = 'same',
                use_bias = False
            )
            return self.conv_sc

    def call(self,x):
        out1 = self.conv1(self.av1(self.bn1(x)))
        out2 = self.conv2(self.av2(self.bn2(out1)))
        out3 = self.conv3(self.av3(self.bn3(out2)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])
        
        return out4


class ResNet(Model):
    def __init__(
        self,
        input_shape,
        output_dim,
    ):
        super().__init__()
        self._layers = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu)
            kl.Conv2D(
                filters = 64,
                kernel_size = 7,
                strides = 2,
                padding = 'same',
                input_shape = input_shape
            ),
            ResBlock(64,256),
            [
                ResBlock(256,256) for _ in range(2)
            ],
            kl.Conv2D(
                512,
                kernel_sie = 1,
                strides = 2,
                use_bias = False
            ),
            [
                ResBlock(512,512) for _ in range(4)
            ]
            kl.Conv2D(
                1024,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                ResBlock(1024,1024) for _ in range(5)
            ],
            kl.GlobalAveragePolling2D(),
            kl.Dense(1000,activation="relu")
            kl.Dense(output_dim,activation = 'linear')
        ]

    def call(self,x):
        for layer in self._layers:
            if isinstance(layer,list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x

class Trainer(object):
    def __init__(self):
        self.model = ResNet(
            input_shape = (128,128,3),
            output_dim = 1
        )
        self.criterion = tf.keras.losses.MSE()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
        self.train_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    

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
        n_batches_train = x_train.shape[0] // batch_size
        n_batches_val = x_val.shape[0] // batch_size

        for epoch in range(epochs):
            x_,t_ = shuffle(x_train,t_train)
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            
            for batch in tqdm(range(n_batches_train)):
                start = batch * batch_size
                end = start + batch_size
                x_batch = x_[start:end]
                t_batch = t_[start:end]
                img_batch = self.get_image(image_path,x_batch)
                self.train_step(img_batch,t_batch)

            for batch in tqdm(range(n_batches_val)):
                start = batch * batch_size
                end = start + batch_size
                x_batch = x_val[start:end]
                t_batch = t_val[start:end]
                img_batch = self.get_image(image_path,x_batch)
                self.val_step(img_batch,t_batch)
            
            self.history['train_loss'].append(self.train_loss.result())
            self.history['val_loss'].append(self.val_loss.result())

            print('epoch {} => train_loss: {},  test_loss: {}'.format(
                epoch + 1,
                self.train_loss.result(),
                self.val_loss.result()
            ))

        fig = plt.figure(figsize = (20,10))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.plot(self.history['train_loss'])
        ax2.plot(self.history['val_loss'])
        ax1.set_title('train_loss')
        ax2.set_title('val_loss')
        plt.show()



    def get_image(
        image_pah,
        x_batch,
        extension = 'jpg'
    ):
        img_batch = list()
        for id in x_batch:
            image = np.load('./img_npy/' + str(id) + '.npy')
            img_batch.append(image)
            
        return np.array(img_batch)

    def train_step(self,x,t):
        with tf.GradientTape() as tape:
            preds = self.model(x)
            loss = self.criterion(t,preds)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradient(zip(grads,model.trainable_variables))
        self.train_loss(loss)

    def val_step(self,x,t):
        preds = self.model(x)
        loss = self.criterion(t,preds)
        self.val_loss(loss)

    def evaluate(self,x_test,t_test,batch_size = 1000):
        loss = tf.metrics.Mean()
        n_batches = x_test.shape[0] // batch_size
        for batch in range(n_batches):
            start = batch_size * batch
            end = start + batch_size
            x_batch = x_test[start:end]
            t_batch = t_test[start:end]
            img_batch = self.get_image(x_batch)
            loss(self.criterion(t_batch,self.model(img_batch)))

        start = batch_size * n_batches
        end = x_test.shape[0]
        x_batch = x_test[start:end]
        t_batch = t_test[start:end]
        img_batch = self.get_image(x_batch)
        loss(self.criterion(t_batch,self.model(img_batch)))
        print(loss.result().numpy())
        return loss.result().numpy()




