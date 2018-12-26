
# coding: utf-8

# ## 卷积神经网络（Convolutional Neural Network, CNN）
# 
# ## 项目：实现一个狗品种识别算法App
# 
# 在这个notebook文件中，有些模板代码已经提供给你，但你还需要实现更多的功能来完成这个项目。除非有明确要求，你无须修改任何已给出的代码。以**'(练习)'**开始的标题表示接下来的代码部分中有你需要实现的功能。这些部分都配有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示。
# 
# 除了实现代码外，你还**需要**回答一些与项目及代码相关的问题。每个需要回答的问题都会以 **'问题 X'** 标记。请仔细阅读每个问题，并且在问题后的 **'回答'** 部分写出完整的答案。我们将根据 你对问题的回答 和 撰写代码实现的功能 来对你提交的项目进行评分。
# 
# >**提示：**Code 和 Markdown 区域可通过 **Shift + Enter** 快捷键运行。此外，Markdown可以通过双击进入编辑模式。
# 
# 项目中显示为_选做_的部分可以帮助你的项目脱颖而出，而不是仅仅达到通过的最低要求。如果你决定追求更高的挑战，请在此 notebook 中完成_选做_部分的代码。
# 
# ---
# 
# ### 让我们开始吧
# 在这个notebook中，你将迈出第一步，来开发可以作为移动端或 Web应用程序一部分的算法。在这个项目的最后，你的程序将能够把用户提供的任何一个图像作为输入。如果可以从图像中检测到一只狗，它会输出对狗品种的预测。如果图像中是一个人脸，它会预测一个与其最相似的狗的种类。下面这张图展示了完成项目后可能的输出结果。（……实际上我们希望每个学生的输出结果不相同！）
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# 在现实世界中，你需要拼凑一系列的模型来完成不同的任务；举个例子，用来预测狗种类的算法会与预测人类的算法不同。在做项目的过程中，你可能会遇到不少失败的预测，因为并不存在完美的算法和模型。你最终提交的不完美的解决方案也一定会给你带来一个有趣的学习经验！
# 
# ### 项目内容
# 
# 我们将这个notebook分为不同的步骤，你可以使用下面的链接来浏览此notebook。
# 
# * [Step 0](#step0): 导入数据集
# * [Step 1](#step1): 检测人脸
# * [Step 2](#step2): 检测狗狗
# * [Step 3](#step3): 从头创建一个CNN来分类狗品种
# * [Step 4](#step4): 使用一个CNN来区分狗的品种(使用迁移学习)
# * [Step 5](#step5): 建立一个CNN来分类狗的品种（使用迁移学习）
# * [Step 6](#step6): 完成你的算法
# * [Step 7](#step7): 测试你的算法
# 
# 在该项目中包含了如下的问题：
# 
# * [问题 1](#question1)
# * [问题 2](#question2)
# * [问题 3](#question3)
# * [问题 4](#question4)
# * [问题 5](#question5)
# * [问题 6](#question6)
# * [问题 7](#question7)
# * [问题 8](#question8)
# * [问题 9](#question9)
# * [问题 10](#question10)
# * [问题 11](#question11)
# 
# 
# ---
# <a id='step0'></a>
# ## 步骤 0: 导入数据集
# 
# ### 导入狗数据集
# 在下方的代码单元（cell）中，我们导入了一个狗图像的数据集。我们使用 scikit-learn 库中的 `load_files` 函数来获取一些变量：
# - `train_files`, `valid_files`, `test_files` - 包含图像的文件路径的numpy数组
# - `train_targets`, `valid_targets`, `test_targets` - 包含独热编码分类标签的numpy数组
# - `dog_names` - 由字符串构成的与标签相对应的狗的种类

# In[21]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# 打印数据统计描述
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# ### 导入人脸数据集
# 
# 在下方的代码单元中，我们导入人脸图像数据集，文件所在路径存储在名为 `human_files` 的 numpy 数组。

# In[22]:


import random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))


# ---
# <a id='step1'></a>
# ## 步骤1：检测人脸
#  
# 我们将使用 OpenCV 中的 [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 来检测图像中的人脸。OpenCV 提供了很多预训练的人脸检测模型，它们以XML文件保存在 [github](https://github.com/opencv/opencv/tree/master/data/haarcascades)。我们已经下载了其中一个检测模型，并且把它存储在 `haarcascades` 的目录中。
# 
# 在如下代码单元中，我们将演示如何使用这个检测模型在样本图像中找到人脸。

# In[23]:


import cv2       # 没有使用命令 pip install opencv-python            
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[7])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()


# 在使用任何一个检测模型之前，将图像转换为灰度图是常用过程。`detectMultiScale` 函数使用储存在 `face_cascade` 中的的数据，对输入的灰度图像进行分类。
# 
# 在上方的代码中，`faces` 以 numpy 数组的形式，保存了识别到的面部信息。它其中每一行表示一个被检测到的脸，该数据包括如下四个信息：前两个元素  `x`、`y` 代表识别框左上角的 x 和 y 坐标（参照上图，注意 y 坐标的方向和我们默认的方向不同）；后两个元素代表识别框在 x 和 y 轴两个方向延伸的长度 `w` 和 `d`。 
# 
# ### 写一个人脸识别器
# 
# 我们可以将这个程序封装为一个函数。该函数的输入为人脸图像的**路径**，当图像中包含人脸时，该函数返回 `True`，反之返回 `False`。该函数定义如下所示。

# In[24]:


# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# ### **【练习】** 评估人脸检测模型

# 
# ---
# 
# <a id='question1'></a>
# ### __问题 1:__ 
# 
# 在下方的代码块中，使用 `face_detector` 函数，计算：
# 
# - `human_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
# - `dog_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
# 
# 理想情况下，人图像中检测到人脸的概率应当为100%，而狗图像中检测到人脸的概率应该为0%。你会发现我们的算法并非完美，但结果仍然是可以接受的。我们从每个数据集中提取前100个图像的文件路径，并将它们存储在`human_files_short`和`dog_files_short`中。

# In[25]:


human_files_short = human_files[:100]
dog_files_short   = train_files[:100]
## 请不要修改上方代码
#print(human_files_short)

## TODO: 基于human_files_short和dog_files_short

list_human = list(map(face_detector, human_files_short))

list_dog   = list(map(face_detector, dog_files_short))

hunman_files_humanper = (list_human.count(True))/len(list_human)

dog_files_humanper    = (list_dog.count(True))/len(list_dog)

#print(list_human)
print("human_files 的前100张图像中，能够检测到人脸的图像占比 {:.2%} ".format(hunman_files_humanper))
#print(list_dog)
print("dog_files 的前100张图像中，能够检测到人脸的图像占比 {:.2%}".format(dog_files_humanper))
## 中的图像测试face_detector的表现


# ---
# 
# <a id='question2'></a>
# 
# ### __问题 2:__ 
# 
# 就算法而言，该算法成功与否的关键在于，用户能否提供含有清晰面部特征的人脸图像。
# 那么你认为，这样的要求在实际使用中对用户合理吗？如果你觉得不合理，你能否想到一个方法，即使图像中并没有清晰的面部特征，也能够检测到人脸？
# 
# __回答:__ 场景是需要进行人脸识别的一些APP或者其他应用上要求用户提供清晰的这个方法还是可行的；在很多场景下这中方法可能不太合适，可以通过训练一个CNN来实现。这里有一篇关于人脸识别算法参考资料：https://www.jiqizhixin.com/articles/2017-12-17-7

# ---
# 
# <a id='Selection1'></a>
# ### 选做：
# 
# 我们建议在你的算法中使用opencv的人脸检测模型去检测人类图像，不过你可以自由地探索其他的方法，尤其是尝试使用深度学习来解决它:)。请用下方的代码单元来设计和测试你的面部监测算法。如果你决定完成这个_选做_任务，你需要报告算法在每一个数据集上的表现。

# In[26]:


## (选做) TODO: 报告另一个面部检测算法在LFW数据集上的表现
### 你可以随意使用所需的代码单元数


# ---
# <a id='step2'></a>
# 
# ## 步骤 2: 检测狗狗
# 
# 在这个部分中，我们使用预训练的 [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 模型去检测图像中的狗。下方的第一行代码就是下载了 ResNet-50 模型的网络结构参数，以及基于 [ImageNet](http://www.image-net.org/) 数据集的预训练权重。
# 
# ImageNet 这目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。它包含超过一千万个 URL，每一个都链接到 [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 中所对应的一个物体的图像。任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果。

# In[27]:


from keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')


# ### 数据预处理
# 
# - 在使用 TensorFlow 作为后端的时候，在 Keras 中，CNN 的输入是一个4维数组（也被称作4维张量），它的各维度尺寸为 `(nb_samples, rows, columns, channels)`。其中 `nb_samples` 表示图像（或者样本）的总数，`rows`, `columns`, 和 `channels` 分别表示图像的行数、列数和通道数。
# 
# 
# - 下方的 `path_to_tensor` 函数实现如下将彩色图像的字符串型的文件路径作为输入，返回一个4维张量，作为 Keras CNN 输入。因为我们的输入图像是彩色图像，因此它们具有三个通道（ `channels` 为 `3`）。
#     1. 该函数首先读取一张图像，然后将其缩放为 224×224 的图像。
#     2. 随后，该图像被调整为具有4个维度的张量。
#     3. 对于任一输入图像，最后返回的张量的维度是：`(1, 224, 224, 3)`。
# 
# 
# - `paths_to_tensor` 函数将图像路径的字符串组成的 numpy 数组作为输入，并返回一个4维张量，各维度尺寸为 `(nb_samples, 224, 224, 3)`。 在这里，`nb_samples`是提供的图像路径的数据中的样本数量或图像数量。你也可以将 `nb_samples` 理解为数据集中3维张量的个数（每个3维张量表示一个不同的图像。

# In[28]:


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# ### 基于 ResNet-50 架构进行预测
# 
# 对于通过上述步骤得到的四维张量，在把它们输入到 ResNet-50 网络、或 Keras 中其他类似的预训练模型之前，还需要进行一些额外的处理：
# 1. 首先，这些图像的通道顺序为 RGB，我们需要重排他们的通道顺序为 BGR。
# 2. 其次，预训练模型的输入都进行了额外的归一化过程。因此我们在这里也要对这些张量进行归一化，即对所有图像所有像素都减去像素均值 `[103.939, 116.779, 123.68]`（以 RGB 模式表示，根据所有的 ImageNet 图像算出）。
# 
# 导入的 `preprocess_input` 函数实现了这些功能。如果你对此很感兴趣，可以在 [这里](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py) 查看 `preprocess_input`的代码。
# 
# 
# 在实现了图像处理的部分之后，我们就可以使用模型来进行预测。这一步通过 `predict` 方法来实现，它返回一个向量，向量的第 i 个元素表示该图像属于第 i 个 ImageNet 类别的概率。这通过如下的 `ResNet50_predict_labels` 函数实现。
# 
# 通过对预测出的向量取用 argmax 函数（找到有最大概率值的下标序号），我们可以得到一个整数，即模型预测到的物体的类别。进而根据这个 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)，我们能够知道这具体是哪个品种的狗狗。
# 

# In[29]:


from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# ### 完成狗检测模型
# 
# 
# 在研究该 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 的时候，你会注意到，狗类别对应的序号为151-268。因此，在检查预训练模型判断图像是否包含狗的时候，我们只需要检查如上的 `ResNet50_predict_labels` 函数是否返回一个介于151和268之间（包含区间端点）的值。
# 
# 我们通过这些想法来完成下方的 `dog_detector` 函数，如果从图像中检测到狗就返回 `True`，否则返回 `False`。

# In[30]:


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# ### 【作业】评估狗狗检测模型
# 
# ---
# 
# <a id='question3'></a>
# ### __问题 3:__ 
# 
# 在下方的代码块中，使用 `dog_detector` 函数，计算：
# 
# - `human_files_short`中图像检测到狗狗的百分比？
# - `dog_files_short`中图像检测到狗狗的百分比？

# In[31]:


### TODO: 测试dog_detector函数在human_files_short和dog_files_short的表现
list_human = list(map(dog_detector, human_files_short))
list_dog = list(map(dog_detector, dog_files_short))
hunman_files_dogper = (list_human.count(True))/len(list_human)
dog_files_dogper = (list_dog.count(True))/len(list_dog)
#print(list_human)
print("human_files_short中图像检测到狗狗的百分比 {:.2%} ".format(hunman_files_dogper))
#print(list_dog)
print("dog_files_short中图像检测到狗狗的百分比 {:.2%}".format(dog_files_dogper))


# ---
# 
# <a id='step3'></a>
# 
# ## 步骤 3: 从头开始创建一个CNN来分类狗品种
# 
# 
# 现在我们已经实现了一个函数，能够在图像中识别人类及狗狗。但我们需要更进一步的方法，来对狗的类别进行识别。在这一步中，你需要实现一个卷积神经网络来对狗的品种进行分类。你需要__从头实现__你的卷积神经网络（在这一阶段，你还不能使用迁移学习），并且你需要达到超过1%的测试集准确率。在本项目的步骤五种，你还有机会使用迁移学习来实现一个准确率大大提高的模型。
# 
# 在添加卷积层的时候，注意不要加上太多的（可训练的）层。更多的参数意味着更长的训练时间，也就是说你更可能需要一个 GPU 来加速训练过程。万幸的是，Keras 提供了能够轻松预测每次迭代（epoch）花费时间所需的函数。你可以据此推断你算法所需的训练时间。
# 
# 值得注意的是，对狗的图像进行分类是一项极具挑战性的任务。因为即便是一个正常人，也很难区分布列塔尼犬和威尔士史宾格犬。
# 
# 
# 布列塔尼犬（Brittany） | 威尔士史宾格犬（Welsh Springer Spaniel）
# - | - 
# <img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# 不难发现其他的狗品种会有很小的类间差别（比如金毛寻回犬和美国水猎犬）。
# 
# 
# 金毛寻回犬（Curly-Coated Retriever） | 美国水猎犬（American Water Spaniel）
# - | -
# <img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">
# 
# 同样，拉布拉多犬（labradors）有黄色、棕色和黑色这三种。那么你设计的基于视觉的算法将不得不克服这种较高的类间差别，以达到能够将这些不同颜色的同类狗分到同一个品种中。
# 
# 黄色拉布拉多犬（Yellow Labrador） | 棕色拉布拉多犬（Chocolate Labrador） | 黑色拉布拉多犬（Black Labrador）
# - | -
# <img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">
# 
# 我们也提到了随机分类将得到一个非常低的结果：不考虑品种略有失衡的影响，随机猜测到正确品种的概率是1/133，相对应的准确率是低于1%的。
# 
# 请记住，在深度学习领域，实践远远高于理论。大量尝试不同的框架吧，相信你的直觉！当然，玩得开心！
# 
# 
# ### 数据预处理
# 
# 
# 通过对每张图像的像素值除以255，我们对图像实现了归一化处理。

# In[32]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 


# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255

valid_tensors = paths_to_tensor(valid_files).astype('float32')/255

test_tensors = paths_to_tensor(test_files).astype('float32')/255


# ### 【练习】模型架构
# 
# 
# 创建一个卷积神经网络来对狗品种进行分类。在你代码块的最后，执行 `model.summary()` 来输出你模型的总结信息。
#     
# 我们已经帮你导入了一些所需的 Python 库，如有需要你可以自行导入。如果你在过程中遇到了困难，如下是给你的一点小提示——该模型能够在5个 epoch 内取得超过1%的测试准确率，并且能在CPU上很快地训练。
# 
# ![Sample CNN](images/sample_cnn.png)

# ---
# 
# <a id='question4'></a>  
# 
# ### __问题 4:__ 
# 
# 在下方的代码块中尝试使用 Keras 搭建卷积网络的架构，并回答相关的问题。
# 
# 1. 你可以尝试自己搭建一个卷积网络的模型，那么你需要回答你搭建卷积网络的具体步骤（用了哪些层）以及为什么这样搭建。
# 2. 你也可以根据上图提示的步骤搭建卷积网络，那么请说明为何如上的架构能够在该问题上取得很好的表现。
# 
# __回答:__ 
# 我决定选择用于图像识别的标准基本CNN结构,该网络以三个卷积层（每一层都有相应的最大池化层）,最后两层由完全连接层组成。第一个卷积层过滤器的数量是16,过滤器对图像中的轮廓很敏感,后面两个卷积层过滤器数量逐渐加倍,以识别更复杂的特征形状。利用（relu）激活函数将使网络不易于线性变换。 'relu'激活函数非常适合图像检测任务,并允许CNN非常有效地进行反向传播。最后一层使用softmax将每个图像的概率应用于每个类标签,并返回概率值。
# 
# 

# In[33]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()  #搭建一个书架

# output_size =1+ (input_size+2*padding-kernel_size)/stride 
### TODO: 定义你的网络架构

model.add(Conv2D(16,(2,2),input_shape=(224,224,3),padding = 'valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(32,(2,2),padding = 'valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(64,(2,2),padding = 'valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(500, activation='relu'))
model.add(Dense(units=133,activation='softmax'))

model.summary()


# In[34]:


## 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# ---

# ## 【练习】训练模型
# 
# 
# ---
# 
# <a id='question5'></a>  
# 
# ### __问题 5:__ 
# 
# 在下方代码单元训练模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。
# 
# 可选题：你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，来优化模型的表现。
# 
# 

# In[35]:


from keras.callbacks import ModelCheckpoint  

### TODO: 设置训练模型的epochs的数量

epochs = 10

### 不要修改下方代码

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# In[36]:


## 加载具有最好验证loss的模型

model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# ### 测试模型
# 
# 在狗图像的测试数据集上试用你的模型。确保测试准确率大于1%。

# In[37]:


# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ---
# <a id='step4'></a>
# ## 步骤 4: 使用一个CNN来区分狗的品种
# 
# 
# 使用 迁移学习（Transfer Learning）的方法，能帮助我们在不损失准确率的情况下大大减少训练时间。在以下步骤中，你可以尝试使用迁移学习来训练你自己的CNN。
# 

# ### 得到从图像中提取的特征向量（Bottleneck Features）

# In[38]:


bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


# ### 模型架构
# 
# 该模型使用预训练的 VGG-16 模型作为固定的图像特征提取器，其中 VGG-16 最后一层卷积层的输出被直接输入到我们的模型。我们只需要添加一个全局平均池化层以及一个全连接层，其中全连接层使用 softmax 激活函数，对每一个狗的种类都包含一个节点。

# In[39]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()


# In[40]:


## 编译模型

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[41]:


## 训练模型
from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


# In[42]:


## 加载具有最好验证loss的模型

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# ### 测试模型
# 现在，我们可以测试此CNN在狗图像测试数据集中识别品种的效果如何。我们在下方打印出测试准确率。

# In[43]:


# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### 使用模型预测狗的品种

# In[44]:


from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]


# ---
# <a id='step5'></a>
# ## 步骤 5: 建立一个CNN来分类狗的品种（使用迁移学习）
# 
# 现在你将使用迁移学习来建立一个CNN，从而可以从图像中识别狗的品种。你的 CNN 在测试集上的准确率必须至少达到60%。
# 
# 在步骤4中，我们使用了迁移学习来创建一个使用基于 VGG-16 提取的特征向量来搭建一个 CNN。在本部分内容中，你必须使用另一个预训练模型来搭建一个 CNN。为了让这个任务更易实现，我们已经预先对目前 keras 中可用的几种网络进行了预训练：
# 
# - [VGG-19](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogVGG19Data.npz) bottleneck features
# - [ResNet-50](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogResnet50Data.npz) bottleneck features
# - [Inception](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogInceptionV3Data.npz) bottleneck features
# - [Xception](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogXceptionData.npz) bottleneck features
# 
# 这些文件被命名为为：
# 
#     Dog{network}Data.npz
# 
# 其中 `{network}` 可以是 `VGG19`、`Resnet50`、`InceptionV3` 或 `Xception` 中的一个。选择上方网络架构中的一个，下载相对应的bottleneck特征，并将所下载的文件保存在目录 `bottleneck_features/` 中。
# 
# 
# ### 【练习】获取模型的特征向量
# 
# 在下方代码块中，通过运行下方代码提取训练、测试与验证集相对应的bottleneck特征。
# 
#     bottleneck_features = np.load('https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogXceptionData.npz')
#     train_{network} = bottleneck_features['train']
#     valid_{network} = bottleneck_features['valid']
#     test_{network} = bottleneck_features['test']

# In[62]:


### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']


# ### 【练习】模型架构
# 
# 建立一个CNN来分类狗品种。在你的代码单元块的最后，通过运行如下代码输出网络的结构：
#     
#         <your model's name>.summary()
#    
# ---
# 
# <a id='question6'></a>  
# 
# ### __问题 6:__ 
# 
# 
# 在下方的代码块中尝试使用 Keras 搭建最终的网络架构，并回答你实现最终 CNN 架构的步骤与每一步的作用，并描述你在迁移学习过程中，使用该网络架构的原因。
# 
# 
# __回答:__ 
# 
# * 第1步：序列模型。
# * 第2步：GlobalAveragePooling2D层。它对Xception预处理的数据采取全局平均池操作。它获得特征映射堆栈，并计算堆栈中每个映射的节点均值，最终变为向量。可以有效的达到减少模型参数的作用，加快训练和拟合时间。
# * 第3步：密集层。该层用于预测，因为狗品种有133个标签，所以设置为133个神经元。并选择softmax功能作为激活函数。
# * 使用Xception的原因：尝试了上述四种模型后Xception准确率最高，准确度达到85.28％。
# * 该网络架构的好处就是充分利用了已经训练好的Xception的特征向量作为输入， Xception网络它由一系列SeparableConv（即“极致的Inception”）、类似ResNet中的残差连接形式和一些其他常规的操作组成。有别于VGG等传统的网络通过堆叠简单的3*3卷积实现特征提取，Xception模块通过组合1*1，3*3，5*5和pooling等结构，用更少的参数和更少的计算开销可以学习到更丰富的特征表示。
# * 相对于早期的CNN模型的准确率不高，该架构通过预训练模型，“迁移”学来的特征，不再从零训练整个结构，只需要针对dense layer进行训练即可得到相对较好的准确率。
# 
# 

# In[63]:


### TODO: 定义你的框架
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dense(133, activation='softmax'))
Xception_model.summary()


# In[64]:


### TODO: 编译模型
Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ---
# 
# ### 【练习】训练模型
# 
# <a id='question7'></a>  
# 
# ### __问题 7:__ 
# 
# 在下方代码单元中训练你的模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。
# 
# 当然，你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 以优化模型的表现，不过这不是必须的步骤。
# 

# In[65]:


### TODO: 训练模型
from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', 
                               verbose=1, save_best_only=True)

Xception_model.fit(train_Xception, train_targets, 
          validation_data=(valid_Xception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


# In[66]:


### TODO: 加载具有最佳验证loss的模型权重
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')


# ---
# 
# ### 【练习】测试模型
# 
# <a id='question8'></a>  
# 
# ### __问题 8:__ 
# 
# 在狗图像的测试数据集上试用你的模型。确保测试准确率大于60%。

# In[67]:


### TODO: 在测试集上计算分类准确率
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ---
# 
# ### 【练习】使用模型测试狗的品种
# 
# 
# 实现一个函数，它的输入为图像路径，功能为预测对应图像的类别，输出为你模型预测出的狗类别（`Affenpinscher`, `Afghan_hound` 等）。
# 
# 与步骤5中的模拟函数类似，你的函数应当包含如下三个步骤：
# 
# 1. 根据选定的模型载入图像特征（bottleneck features）
# 2. 将图像特征输输入到你的模型中，并返回预测向量。注意，在该向量上使用 argmax 函数可以返回狗种类的序号。
# 3. 使用在步骤0中定义的 `dog_names` 数组来返回对应的狗种类名称。
# 
# 提取图像特征过程中使用到的函数可以在 `extract_bottleneck_features.py` 中找到。同时，他们应已在之前的代码块中被导入。根据你选定的 CNN 网络，你可以使用 `extract_{network}` 函数来获得对应的图像特征，其中 `{network}` 代表 `VGG19`, `Resnet50`, `InceptionV3`, 或 `Xception` 中的一个。
#  
# ---
# 
# <a id='question9'></a>  
# 
# ### __问题 9:__

# In[55]:


### TODO: 写一个函数，该函数将图像的路径作为输入
### 然后返回此模型所预测的狗的品种
from extract_bottleneck_features import *

def Xception_predict_breed(img_path):
     # 提取bottleneck特征
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]


# ---
# 
# <a id='step6'></a>
# ## 步骤 6: 完成你的算法
# 
# 
# 
# 实现一个算法，它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：
# 
# - 如果从图像中检测到一只__狗__，返回被预测的品种。
# - 如果从图像中检测到__人__，返回最相像的狗品种。
# - 如果两者都不能在图像中检测到，输出错误提示。
# 
# 我们非常欢迎你来自己编写检测图像中人类与狗的函数，你可以随意地使用上方完成的 `face_detector` 和 `dog_detector` 函数。你__需要__在步骤5使用你的CNN来预测狗品种。
# 
# 下面提供了算法的示例输出，但你可以自由地设计自己的模型！
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# 
# 
# <a id='question10'></a>  
# 
# ### __问题 10:__
# 
# 在下方代码块中完成你的代码。
# 
# ---
# 

# In[56]:


### TODO: 设计你的算法

from keras.preprocessing import image                  
from os import walk
from os import listdir
from os.path import isfile, join
import random


def show_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    plt.imshow(img/255)
    plt.show()
    
### 自由地使用所需的代码单元数吧


# In[87]:


import numpy as np
import cv2

#使用opencv读取文件
def predicts_dog_breed_by_image(path):
    cvimage = cv2.imread(path,1) #参数 -1 代表以原始图像读取，0 灰度模式 1 以RGB格式读取 
    width,height = 224,224 #设置新的图片尺寸 (w,h)
    img_path = cv2.resize(cvimage,(width,height))     # 图像缩放 (224,224)
    
    #cv_rgb = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(img_path)
    
    if (face_detector(path)): # 检测人类
        human_breed = Xception_predict_breed(path)
        print (" 这是一张人类的照片,预测到最亲配狗狗类型 : ", human_breed )
    
    elif (dog_detector(path)==1): #检测狗狗类型
        dog_breed = Xception_predict_breed(path)
        print (" 预测到的狗狗品种是  :", dog_breed )
    else: #否则
        print (" 这张照片无法非人非狗,算法还无法识别 ")


# ---
# <a id='step7'></a>
# ## 步骤 7: 测试你的算法
# 
# 在这个部分中，你将尝试一下你的新算法！算法认为__你__看起来像什么类型的狗？如果你有一只狗，它可以准确地预测你的狗的品种吗？如果你有一只猫，它会将你的猫误判为一只狗吗？
# 
# 
# <a id='question11'></a>  
# 
# ### __问题 11:__
# 
# 在下方编写代码，用至少6张现实中的图片来测试你的算法。你可以使用任意照片，不过请至少使用两张人类图片（要征得当事人同意哦）和两张狗的图片。
# 同时请回答如下问题：
# 
# 1. 输出结果比你预想的要好吗 :) ？或者更糟 :( ？
# 2. 提出至少三点改进你的模型的想法。
# 
# ### 回答:
# 
#  最终模型的测试精度，输出符合我的期望,通过测试6张图片,所有狗和人类都被识别出来,猫的图片也被识别为其他分类,但是同一张图有多个品种的狗的情况下算法识别某一种,这个地方需要改进。
#  ### 改进的想法: 
# 
#  * 1.增加多面部识别,比如多个狗品种合照,分别检测并识别相应的品种
#  * 2.相似度比较高狗品种,增加特定的特征识别  
#  * 3.针对人增加更多特征（人脸识别，仅通过身体部位也可以识别等等）

# In[88]:


## TODO: 在你的电脑上，在步骤6中，至少在6张图片上运行你的算法。
## 自由地使用所需的代码单元数吧
predicts_dog_breed_by_image('images/test_1.jpg')


# In[89]:


predicts_dog_breed_by_image('images/test_2.jpg')


# In[90]:


predicts_dog_breed_by_image('images/test_3.jpg')


# In[91]:


predicts_dog_breed_by_image('images/test_4.jpg')


# In[92]:


predicts_dog_breed_by_image('images/test_5.jpg')


# In[93]:


predicts_dog_breed_by_image('images/test_6.jpg')


# In[94]:


predicts_dog_breed_by_image('images/test_7.jpg')


# **注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出File -> Download as -> HTML (.html)把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。**
