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
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# np.random.seed(2019)
class Data(object):
    def __init__(self,shape_list,
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
        self.shape_list=shape_list
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
            shape=self.shape_list[np.random.choice(range(len(self.shape_list)))]
            degree=[0,90,180,270,360]
            for i in batch_index:
                path,label=self.train_file[i]
                try:
                    img = Image.open(path)
                except:
                    continue
                # img.size 宽 高
                img=img.convert('RGB')
                #数据增强
                ran_degree=np.random.choice(range(len(degree)))
                img=img.rotate(degree[ran_degree],expand=True)
                img=img.resize(size=(shape[1],shape[0]),resample=Image.ANTIALIAS)
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
    def __init__(self,shape_list,batch_size,
                 t_dir='train/truth_pic',
                 f_dir='train/rumor_pic',
                 split_rate=0.1):
        self.shape_list=shape_list
        self.t_dir=t_dir
        self.batch_size=batch_size
        self.f_dir=f_dir
        self.split_rate=split_rate
    def build_network(self,model_name,fine_tune):
        if model_name.lower()=='vgg16':
            if fine_tune:
                base_model=VGG16(include_top=False,input_shape=(None,None,3),pooling='avg')
                for layer in base_model.layers:
                    if layer.name.startswith('block5'):
                        layer.trainable = True
                    else:
                        layer.trainable=False
                x = base_model.output
                x = Dense(1024, activation='relu')(x)
                x=Dropout(0.5)(x)
                x = Dense(512,activation='relu')(x)
                predictions = Dense(1, activation='sigmoid')(x)
                model=Model(base_model.input,predictions)
                model.summary()
                return model
            else:
                base_model = VGG16(include_top=False, input_shape=(None,None,3), pooling='avg')
                for layer in base_model.layers:
                    layer.trainable = True
                x = base_model.output
                x = Dense(1024, activation='relu')(x)
                x = Dropout(0.5)(x)
                x = Dense(512, activation='relu')(x)
                predictions = Dense(1, activation='sigmoid')(x)
                model = Model(base_model.input, predictions)
                model.summary()
                return model
        elif model_name.lower()=='resnet50':
            base_model=ResNet50(include_top=False,input_shape=(None,None,3),pooling='avg',
                                backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
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
        #     x = Dropout(0.5)(x)
        #     x = Dense(1024, activation='relu')(x)
        #     predictions = Dense(1, activation='sigmoid')(x)
        #     model = Model(base_model.input, predictions)
        #     if fine_tune:
        #         for layer in base_model.layers:
        #             layer.trainable = False
        #     model.summary()
        #     return model
        elif model_name.lower()=='resnet101':
            base_model = ResNet101(include_top=False, input_shape=(None,None,3), pooling='avg',
                                   backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='resnet152':
            base_model = ResNet152(include_top=False, input_shape=(None,None,3), pooling='avg',
                                   backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='inceptionresnetv2':
            base_model=InceptionResNetV2(include_top=False,input_shape=(None,None,3),pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='xception':
            base_model=Xception(include_top=False,input_shape=(None,None,3),pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='densenet121':
            base_model = DenseNet121(include_top=False, input_shape=(None,None,3), pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='densenet169':
            base_model = DenseNet169(include_top=False, input_shape=(None,None,3), pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='densenet201':
            base_model = DenseNet201(include_top=False, input_shape=(None,None,3), pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='nasnetlarge':
            base_model = NASNetLarge(include_top=False, input_shape=(None,None,3), pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        elif model_name.lower()=='vgg19':
            base_model = VGG19(include_top=False, input_shape=(None,None,3), pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(base_model.input, predictions)
            if fine_tune:
                for layer in base_model.layers:
                    layer.trainable = False
            model.summary()
            return model
        else:
            base_model = NASNetMobile(include_top=False, input_shape=(None,None,3), pooling='avg')
            x = base_model.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
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
            self.shape_list=[(331,331,3),(662,662,3)]
        elif model_name.lower()=='nasnetmobile':
            self.shape_list=[(224,224,3),(448,448,3)]
        data=Data(shape_list=self.shape_list,batch_size=self.batch_size,true_pic_dir=self.t_dir,
                  rumor_pic_dir=self.f_dir,split_rate=self.split_rate)
        opt=Adam(0.001) if optimizer.lower()=='adam' else SGD(0.001)
        model=self.build_network(model_name=model_name,fine_tune=fine_tune)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc',f1])
        f='fine_tune--' if fine_tune else str()
        model_file_name='multi_scale--'+model_name+'--'+f+optimizer+'-'+'-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}--{val_f1:.5f}.hdf5'
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
        if model_name.lower()=='nasnetlarge':
            self.shape_list=[(331,331,3),(662,662,3)]
        elif model_name.lower()=='nasnetmobile':
            self.shape_list=[(224,224,3),(448,448,3)]
        model=self.build_network(model_name,fine_tune)
        model.load_weights(model_path)
        epoch=model_path.split('--')[0][-3:]
        fail=open('fail_predictions.txt','w',encoding='utf-8')
        c=0
        submit=open('submit_{}_{}.csv'.format(model_name,epoch),'w',encoding='utf-8')
        submit.write('id,label\n')
        id=[]
        label=[]
        probs=[]
        prob_file=open(os.path.join('submissions','multiscale_prob_{}_{}.txt'),'w',encoding='utf-8')
        for i in os.listdir(test_dir):
            c+=1
            try:
                path = os.path.join(test_dir, i)
                img = Image.open(path)
                img=img.convert('RGB')
                prob=[]
                for shape in self.shape_list:
                    img_t = img.resize(size=(shape[1], shape[0]), resample=Image.ANTIALIAS)
                    img_t=np.array(img_t)
                    img_t = (img_t / 255.0) * 2.0 - 1.0
                    inputs=np.expand_dims(img_t,0)
                    pred = model.predict(inputs)
                    prob.append(pred[0,0])
                """
                如果pred<0.5那么应该是虚假图片，按照我的标注应该是0，但是提交要求是虚假新闻是1
                如果pred>=0.5,那么应该是真实图片，真实图片的label是0
                """
                mean_prob=sum(prob)/len(prob)
                probs.append(mean_prob)
                print(i,mean_prob,'{}/{}'.format(c,len(os.listdir(test_dir))))
                prob_file.write('{},{}\n'.format(i.split('.')[0],mean_prob))
                if mean_prob<=0.5:
                    id.append(i.split('.')[0])
                    label.append(1)
                    submit.write('{},{}\n'.format(i.split('.')[0],1))
                else:
                    id.append(i.split('.')[0])
                    label.append(0)
                    submit.write('{},{}\n'.format(i.split('.')[0],0))
            except:
                print('*'*50+i+' ERROR!!!'+'*'*50)
                fail.write(str(i.split('.')[0])+'\n')
        submit.write('\n')
        fail.close()
        submit.close()
        prob_file.close()
        return id,probs
    def ensemble(self,model_path_list,mode='vote',test_image_dir=None):
        """

        :param model_path_list:对于投票制度，必须是奇数个模型
        :param mode: vote,avg,lr,svm
        :return:
        """
        model_name_list=[]
        model_epoch_list=[]
        from collections import defaultdict
        for i in model_path_list:
            temp=i.split('/')[-1].split('_')
            model_name_list.append(temp[0])
            model_epoch_list.append(temp[-1][1:].split('--')[0])
        print(model_name_list)
        print(model_epoch_list)
        csv_name='_'.join([model_name_list[i]+'_'+model_epoch_list[i] for i in range(len(model_name_list))])
        if mode=='vote':
            vote_prediction=open('submissions/vote_{}_predictions.csv'.format(csv_name),
                                 'w',encoding='utf-8')
            predict_dicts=defaultdict(list)
            for i in range(len(model_name_list)):
                temp=open('submit_{}_{}.csv'.format(model_name_list[i], model_epoch_list[i]), 'r',
                     encoding='utf-8').readlines()
                for line in temp:
                    if line=='\n' or line=='id,label\n':
                        pass
                    else:
                        predict_dicts[line.split(',')[0]].append(line.split(',')[-1].strip('\n'))
                        if len(predict_dicts[line.split(',')[0]])>3:
                            raise ValueError('有重复id')
            for i in predict_dicts:
                if predict_dicts[i][0]==predict_dicts[i][2] and predict_dicts[i][0]!=predict_dicts[i][1]:
                    print(i,predict_dicts[i])
            vote_prediction.write('id,label\n')
            for idx in predict_dicts.keys():
                p=predict_dicts[idx]
                one_num=p.count('1')
                if one_num>len(p)-one_num:
                    vote_prediction.write(idx+','+'1\n')
                elif one_num<len(p)-one_num:
                    vote_prediction.write(idx + ',' + '0\n')
                else:
                    raise ValueError('vote模式必须奇数个模型')
            vote_prediction.write('\n')
            vote_prediction.close()
            print('模型投票结束,生成了submissions/vote_{}_predictions.csv文件'.format(csv_name))
        elif mode=='avg':
            files=os.listdir('submissions')
            id2probs = defaultdict(list)
            for i in range(len(model_path_list)):
                prob_file_name='prob_{}_{}.txt'.format(model_name_list[i],model_epoch_list[i])
                if prob_file_name not in files:
                    f=open(os.path.join('submissions',prob_file_name),'w',encoding='utf-8')
                    id, probs = self.predict(model_name=model_name_list[i], model_path=model_path_list[i],
                                             fine_tune=False,
                                             test_dir=test_image_dir)
                    for j in range(len(id)):
                        id2probs[id[j]].append(probs[j])
                        f.write('{},{}\n'.format(id[j],probs[j]))
                        if len(id2probs[id[j]]) > len(model_path_list):
                            raise ValueError(
                                '{}得到的probs是{}个,然而最多只能是{}个'.format(id[j], len(id2probs[id[j]]), len(model_path_list)))
                    f.close()
                else:
                    f=open(os.path.join('submissions',prob_file_name),'r',encoding='utf-8').readlines()
                    print('找到{}'.format(os.path.join('submissions',prob_file_name)))
                    for line in f:
                        if line !='\n':
                            temp = line.split(',')
                            id2probs[temp[0].strip(',')].append(float(temp[-1].strip('\n')))
            f = open('submissions/avg_{}_predictions.csv'.format(csv_name),
                     'w', encoding='utf-8')
            f.write('id,label\n')
            for id in id2probs:
                avg_prob = sum(id2probs[id]) / len(id2probs[id])
                if avg_prob <= 0.5:
                    f.write('{},{}\n'.format(id, 1))
                else:
                    f.write('{},{}\n'.format(id, 0))
            f.write('\n')
            f.close()
            print('模型avg结束,生成了submissions/avg_{}_predictions.csv文件'.format(csv_name))
        else:
            files = os.listdir('submissions')
            id2probs = defaultdict(list)#要预测的输入
            for i in range(len(model_path_list)):
                prob_file_name = 'prob_{}_{}.txt'.format(model_name_list[i], model_epoch_list[i])
                if prob_file_name not in files:
                    f = open(os.path.join('submissions',prob_file_name), 'w', encoding='utf-8')
                    id, probs = self.predict(model_name=model_name_list[i], model_path=model_path_list[i],
                                             fine_tune=False,
                                             test_dir=test_image_dir)
                    for j in range(len(id)):
                        id2probs[id[j]].append(probs[j])
                        f.write('{},{}\n'.format(id[j], probs[j]))
                        if len(id2probs[id[j]]) > len(model_path_list):
                            raise ValueError(
                                '{}得到的probs是{}个,然而最多只能是{}个'.format(id[j], len(id2probs[id[j]]), len(model_path_list)))
                    f.close()
                else:
                    f = open(os.path.join('submissions',prob_file_name), 'r', encoding='utf-8').readlines()
                    print('找到{}'.format(os.path.join('submissions', prob_file_name)))
                    for line in f:
                        if line !='\n':
                            temp = line.split(',')
                            id2probs[temp[0].strip(',')].append(float(temp[-1].strip('\n')))

            f = open('submissions/{}_{}_predictions.csv'.format(mode,csv_name),
                     'w', encoding='utf-8')
            f.write('id,label\n')
            all_train_dict=defaultdict(list)
            all_labels={}
            for name in os.listdir(self.t_dir):
                all_labels[name.split('.')[0]]=1
            for name in os.listdir(self.f_dir):
                all_labels[name.split('.')[0]]=0
            for i in range(len(model_path_list)):
                train_prob_file_name = 'train_prob_{}_{}.txt'.format(model_name_list[i], model_epoch_list[i])
                if train_prob_file_name not in files:
                    train_prob_file = open(os.path.join('submissions',train_prob_file_name), 'w', encoding='utf-8')
                    id, probs = self.predict(model_name=model_name_list[i], model_path=model_path_list[i],
                                             fine_tune=False,
                                             test_dir=self.t_dir)
                    id_1,probs_1=self.predict(model_name=model_name_list[i], model_path=model_path_list[i],
                                             fine_tune=False,
                                             test_dir=self.f_dir)
                    id=id+id_1
                    probs=probs+probs_1
                    for j in range(len(id)):
                        all_train_dict[id[j]].append(probs[j])
                        train_prob_file.write('{},{}\n'.format(id[j], probs[j]))
                        if len(all_train_dict[id[j]]) > len(model_path_list):
                            all_train_dict[id[j]]=all_train_dict[id[j]][:len(model_path_list)]
                            print(
                                '{}得到的probs是{}个,然而最多只能是{}个'.format(id[j], len(all_train_dict[id[j]]), len(model_path_list)))
                    train_prob_file.close()
                else:
                    train_prob_file=open(os.path.join('submissions',train_prob_file_name),'r',encoding='utf-8').readlines()
                    print('找到{}'.format(os.path.join('submissions', train_prob_file_name)))
                    for line in train_prob_file:
                        if line!='\n':
                            temp = line.split(',')
                            all_train_dict[temp[0].strip().strip(',')].append(float(temp[-1].strip('\n')))
            all_data=[]
            for id in all_train_dict:
                all_data.append([id]+list(all_train_dict[id])+list(all_labels[id]))
            all_data=np.array(all_data)
            print('all data shape:{}'.format(all_data.shape))
            np.random.shuffle(all_data)
            from sklearn.model_selection import KFold,train_test_split
            from sklearn.metrics import f1_score
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVC,SVR
            X_train, X_test, y_train, y_test = train_test_split(all_data[:,1:-1], all_data[:,-1], test_size=0.5, random_state=2019)
            print('训练集大小：', X_train.shape, y_train.shape)  # 训练集样本大小
            print('测试集大小：', X_test.shape, y_test.shape)  # 测试集样本大小
            kf=KFold(n_splits=5,shuffle=True)
            model=None
            for k,(train_index,test_index) in enumerate(kf.split(X_train,y_train)):
                kf_x_train=X_train[train_index]
                kf_x_test=X_train[test_index]
                kf_y_train=y_train[train_index]
                kf_y_test=y_train[test_index]
                if mode=='lr':
                    model=LinearRegression()
                elif mode=='svc_linear':
                    model=SVC(kernel='linear')
                elif mode=='svr_linear':
                    model=SVR(kernel='linear')
                elif mode=='svc_rbf':
                    model=SVC()
                elif mode=='svr_rbf':
                    model=SVR()
                else:
                    model=SVC(kernel='rbf')
                model.fit(kf_x_train, kf_y_train)
                print('{}:{}折交叉验证的score是{}'.format(mode,k,model.score(kf_x_test,kf_y_test)))
                prediction=model.predict(kf_x_test)
                for j in prediction:
                    if j not in [0,1]:
                        j=1 if j>=0.5 else 0
                fs=f1_score(kf_y_test,prediction)
                print('{}:{}折交叉验证的f1_score是{}'.format(mode,k,fs))
            cv_pred=model.predict(X_test)
            for j in cv_pred:
                if j not in [0,1]:
                    j=1 if j>=0.5 else 0
            cv_f1_score=f1_score(y_test,cv_pred)
            print('{}:交叉验证的cv_f1_score是{}'.format(mode, cv_f1_score))
            print('开始预测测试集......')
            need_predict_x=np.array(id2probs.values())
            final_prediction=model.predict(need_predict_x)
            for i in range(len(final_prediction)):
                p=0 if final_prediction[i]>0.5 else 1
                f.write(list(id2probs.keys())[i]+','+str(p)+'\n')
            f.write('\n')
            f.close()
            import pickle
            with open('models/{}_{}_{}_emsemble_model.pickle'.format(mode,cv_f1_score,csv_name), 'wb') as f:
                pickle.dump(model, f)
                print('保存{}_{}_{}model完成！'.format(mode,cv_f1_score,csv_name))

if __name__=='__main__':
    m = Mymodel(shape_list=[(256, 256, 3),(128,128,3),(512,512,3)], batch_size=8)
    m.train(model_name='densenet121',fine_tune=False)
    # m.ensemble(
    #     model_path_list=['models/densenet121_adam_-043--0.51932--0.90141.hdf5',
    #                      'models/densenet121_adam_-046--0.78712--0.90434.hdf5',
    #                      'models/Xception_adam_-021--0.53468--0.90346--0.91855.hdf5',
    #                      'models/Xception_adam_-011--0.43643--0.89143--0.90778.hdf5',
    #                      'models/Xception_adam_-024--0.84816--0.90405--0.91918.hdf5',
    #                      'models/densenet121_adam_-074--1.06862--0.91197.hdf5',
    #                      'models/densenet121_adam_-053--0.87286--0.90522.hdf5',
    #                      'models/densenet121_adam_-058--0.89474--0.90933.hdf5'
    #                      ],
    #     mode='lr',test_image_dir='stage1_test'
    # )