# Facial-Expression-Recognition.Pytorch
A CNN based pytorch implementation on facial expression recognition (FER2013 and CK+), achieving 73.112% (state-of-the-art) in FER2013 and 94.64% in CK+ dataset

## Demos ##
![Image text](https://raw.githubusercontent.com/WuJie1010/Facial-Expression-Recognition.Pytorch/master/demo/1.png)
![Image text](https://raw.githubusercontent.com/WuJie1010/Facial-Expression-Recognition.Pytorch/master/demo/2.png)

## Dependencies (Tested by Dai Wei in 2024) ##
- Python 3.8.5
- Pytorch >=0.2.0 (torch 1.5.0+cu101  torchvision 0.6.0+cu101)
- It is difficult to find these versions on the Pytorch official website. The Up owner provides a URL containing the whl files of various old versions:https://download.pytorch.org/whl/cpu/torch_stable.html
- h5py (Preprocessing)
- sklearn (plot confusion matrix)

## Handle environment configuration (Tested by Dai Wei in 2024) ##
Download Anaconda for easy environment management, specific tutorials will not be repeated
```shell script
conda create -n Emojipytorch python=3.8.5
activate Emojipytorch
```
You can download the WHL file from the provided URL on your own, or clone my newly uploaded project, which includes the WHL file corresponding to Windows 64 bit,Torchw file is too large and cannot be uploaded. Please download it yourself
```shell script
#cd enter the project path where the WHL file is located
F：
cd "F:\Latest-Facial-Expression-Recognition-master"
#Install the WHL file corresponding to torch 1.5.0+cu101 torch vision 0.6.0+cu101
pip install torchvision-0.6.0+cpu-cp38-cp38-win_amd64.whl
pip install torch-1.5.0+cpu-cp38-cp38-win_amd64.whl
```
H5py installation
```shell script
pip install h5py
```
sklearn installation：Provide the following URL（ https://www.lfd.uci.edu/ ~Go hlke/Python libs/) Download the corresponding configuration's whl file, or use the whl file under Win 64 in my project
```shell script
F：
cd "F:\Latest-Facial-Expression-Recognition-master"
pip install scikit_learn-1.1.1-cp38-cp38-win_amd64.whl
pip install matplotlib
pip install scikit-image
```
##
至此就完成了全部的项目更新维护工作，本人于2024年亲测有效，并且复现模型，但是由于该模型精度不高，估计是因为提供的训练集过少，如果大家想要进一步研究，建议自己扩充训练集，然后自己跑权重文件，本人复现这个项目也是自己摸爬滚打，其中花费了不少时间解决一些配置过老问题，如果能给我一个Star就更好啦！感谢支持！你的支持是我以后创作并发布的动力！最后祝大家学业进步！
项目还有一些我自己运行过程中的报错，但是每个人的情况可能不同，我就不一一赘述，后续还有什么问题，本文没有提及的可以联系我，我力所能及帮助大家：944899059@qq.com 本人目前大二，时间相对充沛，尽可能回复大家，感谢理解与信赖！
##
##
At this point, all project updates and maintenance work has been completed. I personally tested and validated the model in 2024, and reproduced it. However, due to the low accuracy of the model, it is estimated that the provided training set is too small. If you want to further research, it is recommended to expand the training set yourself, and then run the weight file yourself. I reproduced this project by myself, and spent a lot of time solving some old configuration problems, It would be even better if you could give me a Star! Thank you for your support! Your support is the motivation for me to create and publish in the future! Finally, I wish everyone academic progress!
There are still some errors in the project that I encountered during my own operation, but each person's situation may be different, so I won't go into detail. If there are any further issues that were not mentioned in this article, please feel free to contact me. I will do my best to help everyone: 944899059@qq.com I am currently in my sophomore year and have plenty of time. I will try my best to reply to everyone. Thank you for your understanding and trust!
##

## Errors encountered during the reproduction process (Tested by Dai Wei in 2024) ##
The reproduction of the project was in January 2024. Recently, many friends have seen my messages under the original author's project, contacted me and asked me some questions. After reviewing the engineering logs written by the reproduction project at that time, they decided to share the error problems caused by changes in methods due to the changing times
```shell script
1：There is an incompatibility issue with visualize.py runtime:
    ## Source Code：inputs = Variable(inputs, volatile=True)
    Modify to：
    with torch.no_grad():
    	self.priors = Variable(self.priorbox.forward())
```
```shell script
2：NameError:name 'xrange' is not defined:
  The xrange function in Python 2 becomes range in Python 3, so replace all xranges with range
```
```shell script
3： _, term_width = os.popen('stty size', 'r').read().split() ValueError: not enough values to unpack (expected 2, got 0):
    ## Source Code： _, term_width = os.popen('stty size', 'r').read().split()   term_width = int(term_width)
    Modify to：
    try:
    term_height, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    term_height = int(term_height)
except ValueError:
    term_height = 24
    term_width = 80
```
```shell script
由于在github上整理，本地将不在保存
本地与github差异：github上不具备
torch-1.5.0+cpu-cp38-cp38-win_amd64.whl
FER2013_VGG19文件夹下的PrivateTest_model.t7权重文件
根据报错修改后的代码
```
## Visualize for a test image by a pre-trained model ##
- Firstly, download the pre-trained model from https://drive.google.com/open?id=1Oy_9YmpkSKX1Q8jkOhJbz3Mc7qjyISzU (or https://pan.baidu.com/s/1gCL0TlCwKctAy_5yhzHy5Q,  key: g2d3) and then put it in the "FER2013_VGG19" folder; Next, Put the test image (rename as 1.jpg) into the "images" folder, then 
- python visualize.py

## FER2013 Dataset ##
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

### Preprocessing Fer2013 ###
- first download the dataset(fer2013.csv) then put it in the "data" folder, then
- python preprocess_fer2013.py

### Train and Eval model ###
- python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest

###              fer2013 Accurary             ###

- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>
- Model：   Resnet18 ;     PublicTest_acc：  71.190% ;    PrivateTest_acc：72.973%     

## CK+ Dataset ##
- The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos,
We extracted the last three frames from each sequence in the CK+ dataset, which
contains a total of 981 facial expressions. we use 10-fold Cross validation in the experiment.

### Train and Eval model for a fold ###
- python mainpro_CK+.py --model VGG19 --bs 128 --lr 0.01 --fold 1

### Train and Eval model for all 10 fold ###
- python k_fold_train.py

### plot confusion matrix for all fold ###
- python plot_CK+_confusion_matrix.py --model VGG19

###      CK+ Accurary      ###
- Model：    VGG19 ;       Test_acc：   94.646%   <Br/>
- Model：   Resnet18 ;     Test_acc：   94.040%   

