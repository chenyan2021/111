# 111
This  is  my data science homework

#导入需要的包
import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import matplotlib.pyplot as plt

#参数配置
train_parameters = {
    "input_size": [1, 20, 20],                         
    "class_dim": -1,                                     
    "src_path":"data/data23617/characterData.zip",      
    "target_path":"/home/aistudio/data/dataset",       
    "train_list_path": "./train_data.txt",             
    "eval_list_path": "./val_data.txt",                
    "label_dict":{},                                    
    "readme_path": "/home/aistudio/data/readme.json",   
    "num_epochs": 50,                                    
    "train_batch_size": 16,                            
    "learning_strategy": {                            
        "lr": 0.03                                   
    } 
}

#解压原始数据集——生成数据列表
def unzip_data(src_path,target_path):
       if(not os.path.isdir(target_path)):    
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")

#数据处理
def get_data_list(target_path,train_list_path,eval_list_path):
       class_detail = []
    data_list_path=target_path
    class_dirs = os.listdir(data_list_path)
    if '__MACOSX' in class_dirs:
        class_dirs.remove('__MACOSX')   
    all_class_images = 0
    class_label=0
    class_dim = 0
    trainer_list=[]
    eval_list=[]
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            class_sum = 0
            path = os.path.join(data_list_path,class_dir)
            img_paths = os.listdir(path)
            for img_path in img_paths:          
                if img_path =='.DS_Store':
                    continue
                name_path = os.path.join(path,img_path)                      
                if class_sum % 10 == 0:                                
                    eval_sum += 1                                       
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1 
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")
                class_sum += 1                                         
                all_class_images += 1                                  
            
            class_detail_list['class_name'] = class_dir            
            class_detail_list['class_label'] = class_label         
            class_detail_list['class_eval_images'] = eval_sum       
            class_detail_list['class_trainer_images'] = trainer_sum 
            class_detail.append(class_detail_list)              
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1
            
    train_parameters['class_dim'] = class_dim
    print(train_parameters)   
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image) 
    random.shuffle(trainer_list) 
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image) 
    readjson = {}
    readjson['all_class_name'] = data_list_path                
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'],'w') as f:
        f.write(jsons)
    print ('生成数据列表完成！')

# 自定义data_reader
def data_reader(file_list):
  
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img).astype('float32')
                img = img/255.0
                yield img, int(lab) 
    return reader


#参数初始化
src_path=train_parameters['src_path']
target_path=train_parameters['target_path']
train_list_path=train_parameters['train_list_path']
eval_list_path=train_parameters['eval_list_path']
batch_size=train_parameters['train_batch_size']

unzip_data(src_path,target_path)

with open(train_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
with open(eval_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
     
get_data_list(target_path,train_list_path,eval_list_path)

train_reader = paddle.batch(data_reader(train_list_path),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(data_reader(eval_list_path),
                            batch_size=batch_size,
                            drop_last=True)


#绘制LOSS和ACC
Batch=0
Batchs=[]
all_train_accs=[]
def draw_train_acc(Batchs, train_accs):
    title="training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()

all_train_loss=[]
def draw_train_loss(Batchs, train_loss):
    title="training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()

#定义DNN网络
class MyDNN(fluid.dygraph.Layer):
   
    def __init__(self):
        super(MyDNN,self).__init__()
        self.hidden1 = Linear(20*20,200,act='relu')
        self.hidden2 = Linear(200,150,act='relu')
        self.hidden3 = Linear(150,100,act='relu')
        self.hidden4 = Linear(100,100,act='relu')
        self.out = Linear(100,65,act='softmax')                        
    def forward(self,input):        
        x = fluid.layers.reshape(input,[-1,20*20])
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.hidden3(x)
        x=self.hidden4(x)
        y=self.out(x)
        return y

#训练模型
with fluid.dygraph.guard():
    model=MyDNN() 
    model.train() 
    opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())
    epochs_num=20 
    
    for pass_num in range(epochs_num):
        for batch_id,data in enumerate(train_reader()):
            images=np.array([x[0].reshape(1,20,20) for x in data],np.float32)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)

            predict=model(image) 
            
            loss=fluid.layers.cross_entropy(predict,label)
            
            avg_loss=fluid.layers.mean(loss)
            
            acc=fluid.layers.accuracy(predict,label)
            
            if batch_id!=0 and batch_id%100==0:
                Batch = Batch+100 
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                all_train_accs.append(acc.numpy()[0])
                
                print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
            
            avg_loss.backward()       
            opt.minimize(avg_loss)  
            model.clear_gradients()   
    fluid.save_dygraph(model.state_dict(),'MyDNN')
draw_train_acc(Batchs,all_train_accs)
draw_train_loss(Batchs,all_train_loss)

#模型评估
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model = MyDNN()
    model.load_dict(model_dict) 
    model.eval() 
    for batch_id,data in enumerate(eval_reader()):
        images=np.array([x[0].reshape(1,20,20) for x in data],np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)      
        predict=model(image)     
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)

#处理车牌——二值化、分割出车牌中的每一个字符并保存
license_plate = cv2.imread('work/车牌.png')
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY) 
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY) 
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col]/255
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index-1]
        num += 1
        i = index

characters = []
for i in range(8):
    if i==2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    ndarray = np.pad(binary_plate[:,character_dict[i][0]:character_dict[i][1]], ((0,0), (int(padding), int(padding))), 'constant', constant_values=(0,0))
    ndarray = cv2.resize(ndarray, (20,20))
    cv2.imwrite('work/' + str(i) + '.png', ndarray)
    characters.append(ndarray)
    
def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=False)
    img = img.astype('float32')
    img = img[np.newaxis, ] / 255.0
    return img


#将标签进行转换
print('Label:',train_parameters['label_dict'])
match = {'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N',
        'O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z',
        'yun':'云','cuan':'川','hei':'黑','zhe':'浙','ning':'宁','jin':'津','gan':'赣','hu':'沪','liao':'辽','jl':'吉','qing':'青','zang':'藏',
        'e1':'鄂','meng':'蒙','gan1':'甘','qiong':'琼','shan':'陕','min':'闽','su':'苏','xin':'新','wan':'皖','jing':'京','xiang':'湘','gui':'贵',
        'yu1':'渝','yu':'豫','ji':'冀','yue':'粤','gui1':'桂','sx':'晋','lu':'鲁',
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9'}
L = 0
LABEL ={}
for V in train_parameters['label_dict'].values():
    LABEL[str(L)] = match[V]
    L += 1
print(LABEL)

#构建预测动态图过程
with fluid.dygraph.guard():
    model=MyDNN()
    model_dict,_=fluid.load_dygraph('MyDNN')
    model.load_dict(model_dict)
    model.eval()
    lab=[]
    for i in range(8):
        if i==2:
            continue
        infer_imgs = []
        infer_imgs.append(load_image('work/' + str(i) + '.png'))
        infer_imgs = np.array(infer_imgs)
        infer_imgs = fluid.dygraph.to_variable(infer_imgs)
        result=model(infer_imgs)
        lab.append(np.argmax(result.numpy()))
print(lab)
display(Image.open('work/车牌.png'))
for i in range(len(lab)):
    print(LABEL[str(lab[i])],end='')



                            
