from PIL import Image
import os,numpy as np
from collections import defaultdict
import pandas as pd
"""
defaultdict(<class 'int'>, {'jpg': 19882, 'gif': 471, 'jpeg': 13, 'png': 84, '96acb7a83da068b22297e4f972d1bcb1': 1, 'a765ede36caccb10ac311aa745a9b96a': 1, '28d7eef2921631a91ce76d57182dde64': 1, 'e90088dc9a4b15fbab834ad49338f288': 1, '8e8909d42bd36c629291b3943eebae7a': 1, 'bmp': 1, '29ad0ab7967c3426f806e7cfaeedc803': 1, 'd20c9e59bc8d161896cf510cd048fe7b': 1, '8908e2a96b96bab0dd4f7b4546130383': 1, 'aspx': 1, '32b634458ef43e715744b81b7f92a782': 1})
defaultdict(<class 'int'>, {'jpg': 13523, 'gif': 78, 'c2b22e8b8e9926b640725447ea763207': 1, 'e7a4012e58b922f2425d9917c5e951d0': 1, '68ffa949c420a69b42d688ed4a017a9d': 1, '4bfdd70c758f603c0ef3a978d1c0c012': 1, 'b7120291167b1727c1578d1f9b918fdf': 1, 'aspx': 1, 'ba5a82bf2413b37c56cc9121a8736b7b': 1, '1bbbc0c03c68f044e7b319c5f1973d8e': 1, 'png': 4, '5815bd9efaa7c908f2c15ccd81a87942': 1, '521023883607a8763e153b4e1f02f51d': 1, 'c7c679d8a13f626caf018ed49e3026ee': 1, '336ea3a9f5b003c2f26f933997eada94': 1, 'ebf055dbe26f79bfa8167b88dab31baa': 1, 'dc87d5de47bb54243978c6c4ba38ed74': 1, '933b9f5b8a8be817c44a50fdd5645b3b': 1, 'jpeg': 1, '94fcb35b7a3badbc41ed7ef1f4fb8e60': 1, 'df99d07f2ab3af3ffc094c1cb24bdcfd': 1, 'a88190a3ac4c75c28aaea34cb19336cf': 1, 'a882e6514356d006b8334e6ad7ba8e14': 1, '96acb7a83da068b22297e4f972d1bcb1': 1, 'b43a3b5bc8b89da6a34b083c6ecefb96': 1, '241daec4a7f0b1d6cbed7cd4dd4c07e4': 1, '21be089edb3e181538e5e6d0ed2d32bb': 1, 'c6f55453818abf8f23e7b864ea7b74a1': 1, '1b51b9f4521db9f897bf609e62ee6c47': 1, '2aff75591944b59643de6e56b100b0a8': 1, 'e59e28d8b97b9f0d140d14c0940f012e': 1, '45851620f23d29d5612c87e362b2da47': 1, '9f67a3af3d559d0a1c5731ab505c0ee4': 1})
defaultdict(<class 'int'>, {'jpg': 3773, 'png': 10, 'gif': 50, 'jpeg': 2, '63f1d89eccbd25f4aa2731b3426d99a9': 1, 'f6dcf73eb8b4fd61af2b4e257f288568': 1})
"""
def checktype():
    truth_dir='/media/lishuai/Newsmy/biendata/task2/train/truth_pic'
    rumor_dir='/media/lishuai/Newsmy/biendata/task2/train/rumor_pic'
    test_dir='/media/lishuai/Newsmy/biendata/task2/stage1_test'
    truth_types=defaultdict(int)
    gif,sb,jpeg,png=None,None,None,None
    for i in os.listdir(truth_dir):
        t=i.split('.')[-1]
        if t=='gif':
            gif=os.path.join(truth_dir,i)
        if len(i.split('.'))==1:
            sb=os.path.join(truth_dir,i)
        if t=='jpeg':
            jpeg=os.path.join(truth_dir,i)
        if t=='png':
            png=os.path.join(truth_dir,i)
        truth_types[t]+=1
    rumor_types=defaultdict(int)
    for i in os.listdir(rumor_dir):
        rumor_types[i.split('.')[-1]]+=1
    test_types=defaultdict(int)
    for i in os.listdir(test_dir):
        test_types[i.split('.')[-1]]+=1
    print(truth_types)
    print(rumor_types)
    print(test_types)
    try:
        img=Image.open(gif).convert('RGB')
        img=np.array(img)
        print('gif shape',img.shape)
    except Exception as e:
        print('gif error:{}'.format(e))
    try:
        img = Image.open(jpeg)
        img = np.array(img)
        print('jpeg shape',img.shape)
    except Exception as e:
        print('jpeg error:{}'.format(e))
    try:
        img = Image.open(sb)
        img = np.array(img)
        print('sb shape',img.shape)
    except Exception as e:
        print('sb error:{}'.format(e))
    try:
        img = Image.open(png)
        img = np.array(img)
        print('png shape',img.shape)
    except Exception as e:
        print('png error:{}'.format(e))
def submit(file):
    f=pd.read_csv(file)
    res=pd.DataFrame()
    res['id']=f['id']
    res['label']=f['label']
    res.to_csv(file,index=False,encoding='utf-8')
# submit('submit_densenet121_046.csv')

# tail -fn 50 nohup.out
