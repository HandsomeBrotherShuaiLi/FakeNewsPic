from PIL import Image
import os,numpy as np
from keras.applications import VGG16,VGG19,ResNet50
from keras.applications import InceptionResNetV2,Xception,DenseNet121,DenseNet169,DenseNet201
from keras.applications import NASNetLarge
from keras import Model
from keras.layers import Dense,Dropout
import keras.backend as K
from keras.optimizers import Adam,SGD
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,TensorBoard
K.set_image_data_format('channels_last')
class Data(object):
    def __init__(self,shape,
                 true_pic_dir='F:\\biendata\\task2\\train\\truth_pic',
                 rumor_pic_dir='F:\\biendata\\task2\\train\\rumor_pic',
                 split_rate=0.1,batch_size=16):
        """
        :param shape: (height,width)
        :param true_pic_dir:
        :param rumor_pic_dir:
        :param split_rate:
        :param batch_size:
        """
        self.shape=shape
        self.t_dir=true_pic_dir
        self.f_dir=rumor_pic_dir
        self.train_file=[]
        for i in os.listdir(self.t_dir):
            if i.endswith('.jpg'):
                temp = [os.path.join(self.t_dir, i), 1]
                self.train_file.append(temp)
        for i in os.listdir(self.f_dir):
            if i.endswith('.jpg'):
                temp = [os.path.join(self.f_dir, i), 0]
                self.train_file.append(temp)
        all_index=np.array(range(len(self.train_file)))
        val_num=int(split_rate*len(self.train_file))
        self.val_index=np.random.choice(all_index,size=val_num,replace=False)
        self.train_index=np.array([i for i in all_index if i not in self.val_index])
        np.random.shuffle(self.train_index)
        print('总数是{},训练集{}，验证集{}'.format(len(all_index),len(self.train_index),len(self.val_index)))
        self.batch_size=batch_size
        self.steps_per_epoch=len(self.train_index)//self.batch_size
        self.val_steps_per_epoch=len(self.val_index)//self.batch_size
    def generator(self,is_train=True):
        index=self.train_index if is_train else self.val_index
        start=0
        while True:
            inputs = []
            labels = []
            if start+self.batch_size<len(index):
                batch_index=index[start:start+self.batch_size]
            else:
                batch_index=np.hstack(index[start:],index[:(start+self.batch_size)%len(index)])
            for i in batch_index:
                path,label=self.train_file[i]
                img=Image.open(path)
                # img.size 宽 高
                img=img.resize(size=(self.shape[1],self.shape[0]),resample=Image.ANTIALIAS)
                img=np.array(img)
                img=(img/ 255.0) * 2.0-1.0
                inputs.append(img)
                labels.append([label])
            inputs=np.array(inputs)
            labels=np.array(labels)
            yield inputs,labels
            start=(start+self.batch_size)%len(index)

class Mymodel(object):
    def __init__(self,shape,batch_size,
                 t_dir='train/truth_pic',
                 f_dir='train/rumor_pic',
                 split_rate=0.1):
        self.shape=shape
        self.t_dir=t_dir
        self.batch_size=batch_size
        self.f_dir=f_dir
        self.split_rate=split_rate
    def build_network(self,model_name,fine_tune):
        if model_name.lower()=='vgg16':
            if fine_tune:
                base_model=VGG16(include_top=False,input_shape=self.shape,pooling='avg')
                for layer in base_model.layers:
                    if layer.name.startswith('block5'):
                        layer.trainable = True
                    else:
                        layer.trainable=False
                x = base_model.output
                x = Dense(1024, activation='relu')(x)
                x=Dropout(0.3)(x)
                x = Dense(512,activation='relu')(x)
                predictions = Dense(1, activation='sigmoid')(x)
                model=Model(base_model.input,predictions)
                model.summary()
                return model
            else:
                base_model = VGG16(include_top=False, input_shape=self.shape, pooling='avg')
                for layer in base_model.layers:
                    layer.trainable = True
                x = base_model.output
                x = Dense(1024, activation='relu')(x)
                x = Dropout(0.3)(x)
                x = Dense(512, activation='relu')(x)
                predictions = Dense(1, activation='sigmoid')(x)
                model = Model(base_model.input, predictions)
                model.summary()
                return model
    def train(self,model_name,fine_tune,optimizer='adam'):
        data=Data(shape=self.shape,batch_size=self.batch_size,true_pic_dir=self.t_dir,
                  rumor_pic_dir=self.f_dir,split_rate=self.split_rate)
        opt=Adam(0.001) if optimizer.lower()=='adam' else SGD(0.001)
        model=self.build_network(model_name=model_name,fine_tune=fine_tune)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])
        f='fine_tune_' if fine_tune else str()
        model_file_name=model_name+'_'+f+optimizer+'_'+'-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'
        model.fit_generator(
            generator=data.generator(is_train=True),
            steps_per_epoch=data.steps_per_epoch,
            validation_data=data.generator(is_train=False),
            validation_steps=data.val_steps_per_epoch,
            verbose=1,
            initial_epoch=0,epochs=100,
            callbacks=[
                TensorBoard('logs'),
                ReduceLROnPlateau(monitor='val_acc',patience=7,verbose=1),
                EarlyStopping(monitor='val_acc',patience=40,verbose=1),
                ModelCheckpoint(filepath=os.path.join('models',model_file_name),monitor='val_acc',
                                verbose=1,save_weights_only=False,save_best_only=True)
            ]
        )


m=Mymodel(shape=(512,512,3),batch_size=16)
m.train(model_name='vgg16',fine_tune=True,optimizer='adam')

