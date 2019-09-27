import os,numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
from keras.applications import VGG16,VGG19
from keras.applications import InceptionResNetV2,Xception,DenseNet121,DenseNet169,DenseNet201
from keras.applications import NASNetLarge,NASNetMobile
from keras import Model
from keras_applications.resnet import ResNet50,ResNet152,ResNet101
from keras.layers import Dense,Dropout
import keras.backend as K
import keras
from keras.optimizers import Adam,SGD
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,TensorBoard
K.set_image_data_format('channels_last')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# np.random.seed(2019)
class Data(object):
    def __init__(self,shape,
                 true_pic_dir='F:\\biendata\\task2\\train\\truth_pic',
                 rumor_pic_dir='F:\\biendata\\task2\\train\\rumor_pic',
                 split_rate=0.1,batch_size=16):
        """
        need to re-train
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
            temp = [os.path.join(self.t_dir, i), 1]
            self.train_file.append(temp)
        for i in os.listdir(self.f_dir):
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
                batch_index=np.hstack((index[start:],index[:(start+self.batch_size)%len(index)]))
            for i in batch_index:
                path,label=self.train_file[i]
                try:
                    img = Image.open(path)
                except:
                    continue
                # img.size 宽 高
                img=img.convert('RGB')
                img=img.resize(size=(self.shape[1],self.shape[0]),resample=Image.ANTIALIAS)
                img=np.array(img)
                img=(img/ 255.0) * 2.0-1.0
                inputs.append(img)
                labels.append([label])
            if len(inputs)<self.batch_size:
                temp_index=np.array(range(len(inputs)))
                rest_num=self.batch_size-len(inputs)
                r_index=np.random.choice(temp_index,rest_num)
                rest_inputs=[inputs[j] for j in r_index]
                rest_labels=[labels[j] for j in r_index]
                inputs+=rest_inputs
                labels+=rest_labels
            inputs=np.array(inputs)
            labels=np.array(labels)
            # print(inputs.shape)
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
        elif model_name.lower()=='resnet50':
            base_model=ResNet50(include_top=False,input_shape=self.shape,pooling='avg',
                                backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable=False
            model.summary()
            return model
        # elif model_name.lower()=='resnet34':
        #     base_model=ResNet34(include_top=False, input_shape=self.shape, pooling='avg')
        #     x = base_model.output
        #     x = Dense(1024, activation='relu')(x)
        #     x = Dropout(0.3)(x)
        #     x = Dense(1024, activation='relu')(x)
        #     predictions = Dense(1, activation='sigmoid')(x)
        #     model = Model(base_model.input, predictions)
        #     if fine_tune:
        #         for layer in base_model.layers:
        #             layer.trainable = False
        #     model.summary()
        #     return model
        elif model_name.lower()=='resnet101':
            base_model = ResNet101(include_top=False, input_shape=self.shape, pooling='avg',
                                   backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='resnet152':
            base_model = ResNet152(include_top=False, input_shape=self.shape, pooling='avg',
                                   backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='inceptionresnetv2':
            base_model=InceptionResNetV2(include_top=False,input_shape=self.shape,pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='xception':
            base_model=Xception(include_top=False,input_shape=self.shape,pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='densenet121':
            base_model = DenseNet121(include_top=False, input_shape=self.shape, pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='densenet169':
            base_model = DenseNet169(include_top=False, input_shape=self.shape, pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='densenet201':
            base_model = DenseNet201(include_top=False, input_shape=self.shape, pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='nasnetlarge':
            base_model = NASNetLarge(include_top=False, input_shape=self.shape, pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='vgg19':
            base_model = VGG19(include_top=False, input_shape=self.shape, pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        else:
            base_model = NASNetMobile(include_top=False, input_shape=self.shape, pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model

    def train(self,model_name,fine_tune,optimizer='adam'):
        if model_name.lower()=='nasnetlarge':
            self.shape=(331,331,3)
        elif model_name.lower()=='nasnetmobile':
            self.shape=(224,224,3)
        data=Data(shape=self.shape,batch_size=self.batch_size,true_pic_dir=self.t_dir,
                  rumor_pic_dir=self.f_dir,split_rate=self.split_rate)
        opt=Adam(0.001) if optimizer.lower()=='adam' else SGD(0.001)
        model=self.build_network(model_name=model_name,fine_tune=fine_tune)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])
        # model.load_weights('models/inceptionresnetv2_adam_-026--0.37283--0.85329.hdf5')
        f='fine_tune_' if fine_tune else str()
        model_file_name=model_name+'_'+f+optimizer+'_'+'-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'
        model.fit_generator(
            generator=data.generator(is_train=True),
            steps_per_epoch=data.steps_per_epoch,
            validation_data=data.generator(is_train=False),
            validation_steps=data.val_steps_per_epoch,
            verbose=1,
            initial_epoch=0,epochs=200,
            callbacks=[
                TensorBoard('logs'),
                ReduceLROnPlateau(monitor='val_acc',patience=7,verbose=1),
                EarlyStopping(monitor='val_acc',patience=40,verbose=1,restore_best_weights=True),
                ModelCheckpoint(filepath=os.path.join('models',model_file_name),monitor='val_acc',
                                verbose=1,save_weights_only=False,save_best_only=True)
            ]
        )
    def predict(self,model_path,test_dir,fine_tune,model_name='vgg16'):
        """
        对于虚假新闻图片检测任务，id是图片id，label是该图片对应的预测结果，int类型，取值范围为{0,1}，
        虚假新闻图片为1，真实新闻图片为0；
        :param model_path:
        :param test_dir:
        :param fine_tune:
        :param model_name:
        :return:
        """
        import pandas as pd
        if model_name.lower()=='nasnetlarge':
            self.shape=(331,331,3)
        elif model_name.lower()=='nasnetmobile':
            self.shape=(224,224,3)
        model=self.build_network(model_name,fine_tune)
        model.load_weights(model_path)
        epoch=model_path.split('--')[0][-3:]
        fail=open('fail_predictions.txt','w',encoding='utf-8')
        c=0
        submit=pd.DataFrame()
        id=[]
        label=[]
        for i in os.listdir(test_dir):
            c+=1
            try:
                path = os.path.join(test_dir, i)
                img = Image.open(path)
                img=img.convert('RGB')
                img = img.resize(size=(self.shape[1], self.shape[0]), resample=Image.ANTIALIAS)
                img = np.array(img)
                img = (img / 255.0) * 2.0 - 1.0
                inputs=np.expand_dims(img,axis=0)
                pred=model.predict(inputs)
                """
                如果pred<0.5那么应该是虚假图片，按照我的标注应该是0，但是提交要求是虚假新闻是1
                如果pred>=0.5,那么应该是真实图片，真实图片的label是0
                """
                print(i,pred[0][0],'{}/{}'.format(c,len(os.listdir(test_dir))))
                if pred[0][0]<=0.5:
                    id.append(i.split('.')[0])
                    label.append(1)
                else:
                    id.append(i.split('.')[0])
                    label.append(0)
            except:
                print('*'*50+i+' ERROR!!!'+'*'*50)
                fail.write(str(i.split('.')[0])+'\n')
        submit['id']=id
        submit['label']=label
        submit.to_csv('submit_{}_{}.csv'.format(model_name,epoch),index=False)
        fail.close()

if __name__=='__main__':
    m = Mymodel(shape=(256, 256, 3), batch_size=16)
    # m.train(model_name='resnet101', fine_tune=False, optimizer='adam')
    m.predict(model_name='densenet121',
              fine_tune=False,test_dir='/media/lishuai/Newsmy/biendata/task2/stage1_test',
              model_path='models/densenet121_adam_-057--0.85544--0.90698.hdf5')
    m.predict(model_name='densenet121',
              fine_tune=False, test_dir='/media/lishuai/Newsmy/biendata/task2/stage1_test',
              model_path='models/densenet121_adam_-058--0.89474--0.90933.hdf5')