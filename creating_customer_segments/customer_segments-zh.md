# 机器学习工程师纳米学位
## 非监督式学习
## 项目：创建客户细分

欢迎来到机器学习工程师纳米学位的第三个实战项目！在此 notebook 中，我们已经为你提供了一些模板代码，你需要实现其他必要功能，以便成功地完成此项目。以**实现**开头的部分表示你必须为下面的代码块提供额外的功能。我们将在每部分提供说明，并在代码块中用 `'TODO'` 语句标记具体的实现要求。请务必仔细阅读说明！

除了实现代码之外，你必须回答一些问题，这些问题与项目和你的实现有关。每个部分需要回答的问题都在开头以**问题 X** 标记。请仔细阅读每个问题并在下面以**答案：**开头的文本框中提供详细的答案。我们将根据你的每个问题答案和所提供的实现代码评估你提交的项目。

>**注意：**你可以使用键盘快捷键 **Shift + Enter** 执行代码和 Markdown 单元格。此外，可以通过双击进入编辑模式，编辑 Markdown 单元格。

## 开始

在此项目中，你将分析一个数据集，该数据集包含关于来自多种产品类别的各种客户年度消费额（*货币单位*计价）的数据。该项目的目标之一是准确地描述与批发商进行交易的不同类型的客户之间的差别。这样可以使分销商清晰地了解如何安排送货服务，以便满足每位客户的需求。

你可以在 [UCI 机器学习代码库](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)中找到此项目的数据集。对于此项目，我们将忽略特征 `'Channel'` 和 `'Region'`，重点分析记录的六个客户产品类别。

运行以下代码块，以加载批发客户数据集以及几个此项目所需的必要 Python 库。你可以根据系统报告的数据集大小判断数据集是否已成功加载。


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
```

    Wholesale customers dataset has 440 samples with 6 features each.
    

## 数据探索
在此部分，你将通过可视化图表和代码开始探索数据，并了解每个特征相互之间的关系。你将观察数据集的统计学描述内容，考虑每个特征之间的联系，从数据集中选择几个样本数据集并在整个项目期间跟踪这几个样本。

运行以下代码块，以观察数据集的统计学描述内容。注意数据集由  6 个重要的产品类别构成：**“Fresh”**、**“Milk”**、**“Grocery”**、**“Frozen”**、**“Detergents_Paper”**和**“Delicatessen”**。思考每个类别代表你可以购买的哪些产品。


```python
# Display a description of the dataset
display(data.describe())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>


### 实现：选择样本
为了更好地通过分析了解客户以及他们的数据会如何变化，最好的方式是选择几个样本数据点并更详细地分析这些数据点。在以下代码块中，向 `indices` 列表中添加**三个**你所选的索引，表示将跟踪的客户。建议尝试不同的样本集合，直到获得相互之间差异很大的客户。


```python
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [60,100,380]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)
```

    Chosen samples of wholesale customers dataset:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8590</td>
      <td>3045</td>
      <td>7854</td>
      <td>96</td>
      <td>4095</td>
      <td>225</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11594</td>
      <td>7779</td>
      <td>12144</td>
      <td>3252</td>
      <td>8035</td>
      <td>3029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28257</td>
      <td>944</td>
      <td>2146</td>
      <td>3881</td>
      <td>600</td>
      <td>270</td>
    </tr>
  </tbody>
</table>
</div>



```python
benchmark = data.mean()
((samples-benchmark) / benchmark).plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x234812d7320>




    
![png](customer_segments-zh_files/customer_segments-zh_6_1.png)
    



```python
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
log_data = np.log(samples)
log_samples = np.log(log_data)
pca = PCA(n_components=6)
pca.fit(log_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(log_data, pca)
```


    
![png](customer_segments-zh_files/customer_segments-zh_7_0.png)
    


### 问题 1
查看样本客户对应的每个产品类别的总购买成本和上述统计学描述内容。  

* 你所选的每个样本可以代表什么样的（客户）场所？

**提示：**场所示例包括市场、咖啡厅、熟食店、零售店等地点。避免使用具体的名称，例如在将样本客户描述为餐厅时使用“麦当劳”。你可以使用参考均值与你的样本进行比较。均值如下所示：

* Fresh：12000.2977
* Milk：5796.2
* Grocery:7951.28
* Frozen：3071.9
* Detergents_paper：2881.4
* Delicatessen：1524.8

知道这些均值后，你的样本比较结果如何？有助于你确定他们可能属于哪种场所吗？


**答案：**



* 样本0：Detergents_Paper在75%，Fresh、Milk接近中位数、Grocery接近均值、 Delicatessen和Frozen比较少，推断是零售店; 

* 样本1：Detergents_Paper、Fresh、Milk、Grocery、 Delicatessen和Frozen所有特征都在均值以上，但是没有哪样特别多，推断是大型超市; 

* 样本2：Fresh超过75%、Frozen为均值附近、其他特征较少，推测是新鲜肉菜市场;

### 实现：特征相关性
一个值得考虑的有趣问题是，在六个产品类别中是否有一个（或多个）类别实际上在了解客户购买情况时相互有关联性。也就是说，是否能够判断购买一定量的某个类别产品的客户也一定会购买数量成比例的其他类别的产品？我们可以通过以下方式轻松地做出这一判断：删除某个特征，并用一部分数据训练监督式回归学习器，然后对模型评估所删除特征的效果进行评分。

你需要在下面的代码块中实现以下步骤：
 - 通过使用 `DataFrame.drop` 函数删除你所选的特征，为 `new_data` 分配一个数据副本。
 - 使用 `sklearn.cross_validation.train_test_split` 将数据集拆分为训练集和测试集。
   - 使用删除的特征作为目标标签。将 `test_size` 设为 `0.25` 并设置 `random_state`。
 - 导入决策树回归器，设置 `random_state`，并将学习器拟合到训练数据中。
 - 使用回归器 `score` 函数报告测试集的预测分数。


```python
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
from sklearn.metrics import accuracy_score
from sklearn import tree
y = data['Detergents_Paper']
new_data = data.copy()

new_data.drop(['Detergents_Paper'], axis = 1, inplace = True)


# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, y,test_size=.25,random_state=50)

# TODO: Create a decision tree regressor and fit it to the training set

regressor =  tree.DecisionTreeRegressor(random_state=10)
regressor.fit(X_train,y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)
print("预测分数:",score)
```

    预测分数: 0.8017669397135979
    

### 问题 2

* 你尝试预测的是哪个特征？
* 报告的预测分数是多少？
* 该特征对确定客户的消费习惯有影响吗？

**提示：**确定系数 `R^2` 的范围是 0 到 1，1 表示完美拟合。负的 `R^2` 表示模型无法拟合数据。如果特定特征的分数很低，则表明使用其他特征很难预测该特征点，因此在考虑相关性时这个特征很重要。

**答案：**

#### 1：你尝试预测的是哪个特征？
**回答：** Detergents_Paper
#### 2：报告的预测分数是多少？
**回答：**  0.801766939714
#### 3：该特征对确定客户的消费习惯有影响吗？
**回答：** Detergents_Paper=0.801拟合性表现很好；说明这个特征与其他特征相关性不重要；对消费习惯有一定影响，但可以用其他特征体现。
### 可视化特征分布图
为了更好地理解数据集，我们可以为数据中的六个产品特征分别构建一个散布矩阵。如果你发现你在上面尝试预测的特征与识别特定客户有关，那么下面的散布矩阵可能会显示该特征与其他特征之间没有任何关系。相反，如果你认为该特征与识别特定客户不相关，散布矩阵可能会显示该特征与数据中的另一个特征有关系。运行以下代码块，以生成散布矩阵。


```python
# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```


    
![png](customer_segments-zh_files/customer_segments-zh_11_0.png)
    



```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.000000</td>
      <td>0.100510</td>
      <td>-0.011854</td>
      <td>0.345881</td>
      <td>-0.101953</td>
      <td>0.244690</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>0.100510</td>
      <td>1.000000</td>
      <td>0.728335</td>
      <td>0.123994</td>
      <td>0.661816</td>
      <td>0.406368</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-0.011854</td>
      <td>0.728335</td>
      <td>1.000000</td>
      <td>-0.040193</td>
      <td>0.924641</td>
      <td>0.205497</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>0.345881</td>
      <td>0.123994</td>
      <td>-0.040193</td>
      <td>1.000000</td>
      <td>-0.131525</td>
      <td>0.390947</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-0.101953</td>
      <td>0.661816</td>
      <td>0.924641</td>
      <td>-0.131525</td>
      <td>1.000000</td>
      <td>0.069291</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>0.244690</td>
      <td>0.406368</td>
      <td>0.205497</td>
      <td>0.390947</td>
      <td>0.069291</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.cov()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.599549e+08</td>
      <td>9.381789e+06</td>
      <td>-1.424713e+06</td>
      <td>2.123665e+07</td>
      <td>-6.147826e+06</td>
      <td>8.727310e+06</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>9.381789e+06</td>
      <td>5.446997e+07</td>
      <td>5.108319e+07</td>
      <td>4.442612e+06</td>
      <td>2.328834e+07</td>
      <td>8.457925e+06</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-1.424713e+06</td>
      <td>5.108319e+07</td>
      <td>9.031010e+07</td>
      <td>-1.854282e+06</td>
      <td>4.189519e+07</td>
      <td>5.507291e+06</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>2.123665e+07</td>
      <td>4.442612e+06</td>
      <td>-1.854282e+06</td>
      <td>2.356785e+07</td>
      <td>-3.044325e+06</td>
      <td>5.352342e+06</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-6.147826e+06</td>
      <td>2.328834e+07</td>
      <td>4.189519e+07</td>
      <td>-3.044325e+06</td>
      <td>2.273244e+07</td>
      <td>9.316807e+05</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>8.727310e+06</td>
      <td>8.457925e+06</td>
      <td>5.507291e+06</td>
      <td>5.352342e+06</td>
      <td>9.316807e+05</td>
      <td>7.952997e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### 问题 3
* 将散布矩阵作为参考，讨论数据集的分布情况，尤其是正态性、离群值、大量接近 0 的数据点等。如果你需要区分某些图表，以便进一步阐述你的观点，也可以这么做。
* 有任何特征对存在某种联系吗？
* 能够佐证你尝试预测的特征存在相关性论点吗？
* 这些特征的数据分布情况如何？

**提示：**数据是正态分布的吗？ 大部分数据点都位于哪个位置？你可以使用 [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) 得出特征关系，然后使用 [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)（要提供给热图的数据是联系值。例如 `data.corr()`）可视化这些特征，以进一步获得信息。

**答案：**
Grocery和Detergents_Paper相关性强；Grocery和Milk也有一定的相关性。具体可以通过data.corr()显示的数据，越接近1的相关性越强；
数据的分布不是正态性，大部分数据集中在8000以内，离群的数据只有一小部分；

## 数据预处理
在此部分，你将预处理数据（对数据进行缩放并检测离群值，或许还会删除离群值），以便更好地表示客户数据。预处理数据通常是很关键的步骤，可以确保通过分析获得的结果有显著统计意义。

### 实现：特征缩放
如果数据不是正态分布数据，尤其是如果均值和中值差别很大（表明非常偏斜），通常[比较合适的方法](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics)是应用非线性缩放——尤其是对金融数据来说。实现这种缩放的一种方式是采用[博克斯-卡克斯检定](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html)，该检定会计算能缩小偏斜情况的最佳次方转换方式。适合大多数情况的更简单方式是采用自然对数。

你需要在下面的代码块中实现以下步骤：
 - 通过应用对数缩放将数据副本赋值给 `log_data`。你可以使用 `np.log` 函数完成这一步。
 - 在应用对数缩放后，将样本数据副本赋值给 `log_samples`。同样使用 `np.log`。


```python
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```


    
![png](customer_segments-zh_files/customer_segments-zh_16_0.png)
    


### 观察
在对数据应用自然对数缩放后，每个特征的分布应该看起来很正态了。对于你之前可能发现相互有关联的任何特征对，在此部分观察这种联系是否依然存在（是否比之前更明显）。

运行以下代码，看看在应用自然对数后样本数据有何变化。


```python
# Display the log-transformed sample data
display(log_samples)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.058354</td>
      <td>8.021256</td>
      <td>8.968778</td>
      <td>4.564348</td>
      <td>8.317522</td>
      <td>5.416100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.358243</td>
      <td>8.959183</td>
      <td>9.404590</td>
      <td>8.087025</td>
      <td>8.991562</td>
      <td>8.015988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.249096</td>
      <td>6.850126</td>
      <td>7.671361</td>
      <td>8.263848</td>
      <td>6.396930</td>
      <td>5.598422</td>
    </tr>
  </tbody>
</table>
</div>


### 实现：检测离群值
对于任何分享的数据预处理步骤来说，检测数据中的离群值都极为重要。如果结果考虑了离群值，那么这些离群值通常都会使结果出现偏斜。在判断什么样的数据属于离群值时，可以采用很多“一般规则”。在此项目中，我们将使用 [Tukey 方法检测离群值](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/)：*离群值步长*等于 1.5 倍四分位距 (IQR)。如果某个数据点的特征超出了该特征的离群值步长范围，则该特征属于异常特征。

你需要在下面的代码块中实现以下步骤：
 - 将给定特征的第 25 百分位值赋值给 `Q1`。 为此，请使用 `np.percentile`。
 - 将给定特征的第 75 百分位值赋值给 `Q3`。同样使用 `np.percentile`。
 - 将给定特征的离群值步长计算结果赋值给 `step`。
 - （可选步骤）通过向 `outliers` 列表添加索引，从数据集中删除某些数据点。

**注意：**如果你选择删除任何离群值，确保样本数据不包含任何此类数据点！  
实现这一步骤后，数据集将存储在变量 `good_data` 中。


```python
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    f_s = log_data[feature]
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 =np.percentile(f_s,25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(f_s,75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step =  1.5 * (Q3 - Q1)
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [65,66,75,128,154]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
```

    Data points considered outliers for the feature 'Fresh':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.389072</td>
      <td>9.163249</td>
      <td>9.575192</td>
      <td>5.645447</td>
      <td>8.964184</td>
      <td>5.049856</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.098612</td>
      <td>7.979339</td>
      <td>8.740657</td>
      <td>6.086775</td>
      <td>5.407172</td>
      <td>6.563856</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.135494</td>
      <td>7.869402</td>
      <td>9.001839</td>
      <td>4.976734</td>
      <td>8.262043</td>
      <td>5.379897</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>171</th>
      <td>5.298317</td>
      <td>10.160530</td>
      <td>9.894245</td>
      <td>6.478510</td>
      <td>9.079434</td>
      <td>8.740337</td>
    </tr>
    <tr>
      <th>193</th>
      <td>5.192957</td>
      <td>8.156223</td>
      <td>9.917982</td>
      <td>6.865891</td>
      <td>8.633731</td>
      <td>6.501290</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2.890372</td>
      <td>8.923191</td>
      <td>9.629380</td>
      <td>7.158514</td>
      <td>8.475746</td>
      <td>8.759669</td>
    </tr>
    <tr>
      <th>304</th>
      <td>5.081404</td>
      <td>8.917311</td>
      <td>10.117510</td>
      <td>6.424869</td>
      <td>9.374413</td>
      <td>7.787382</td>
    </tr>
    <tr>
      <th>305</th>
      <td>5.493061</td>
      <td>9.468001</td>
      <td>9.088399</td>
      <td>6.683361</td>
      <td>8.271037</td>
      <td>5.351858</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1.098612</td>
      <td>5.808142</td>
      <td>8.856661</td>
      <td>9.655090</td>
      <td>2.708050</td>
      <td>6.309918</td>
    </tr>
    <tr>
      <th>353</th>
      <td>4.762174</td>
      <td>8.742574</td>
      <td>9.961898</td>
      <td>5.429346</td>
      <td>9.069007</td>
      <td>7.013016</td>
    </tr>
    <tr>
      <th>355</th>
      <td>5.247024</td>
      <td>6.588926</td>
      <td>7.606885</td>
      <td>5.501258</td>
      <td>5.214936</td>
      <td>4.844187</td>
    </tr>
    <tr>
      <th>357</th>
      <td>3.610918</td>
      <td>7.150701</td>
      <td>10.011086</td>
      <td>4.919981</td>
      <td>8.816853</td>
      <td>4.700480</td>
    </tr>
    <tr>
      <th>412</th>
      <td>4.574711</td>
      <td>8.190077</td>
      <td>9.425452</td>
      <td>4.584967</td>
      <td>7.996317</td>
      <td>4.127134</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Milk':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>10.039983</td>
      <td>11.205013</td>
      <td>10.377047</td>
      <td>6.894670</td>
      <td>9.906981</td>
      <td>6.805723</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.220590</td>
      <td>4.718499</td>
      <td>6.656727</td>
      <td>6.796824</td>
      <td>4.025352</td>
      <td>4.882802</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>356</th>
      <td>10.029503</td>
      <td>4.897840</td>
      <td>5.384495</td>
      <td>8.057377</td>
      <td>2.197225</td>
      <td>6.306275</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Grocery':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Frozen':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>8.431853</td>
      <td>9.663261</td>
      <td>9.723703</td>
      <td>3.496508</td>
      <td>8.847360</td>
      <td>6.070738</td>
    </tr>
    <tr>
      <th>57</th>
      <td>8.597297</td>
      <td>9.203618</td>
      <td>9.257892</td>
      <td>3.637586</td>
      <td>8.932213</td>
      <td>7.156177</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>145</th>
      <td>10.000569</td>
      <td>9.034080</td>
      <td>10.457143</td>
      <td>3.737670</td>
      <td>9.440738</td>
      <td>8.396155</td>
    </tr>
    <tr>
      <th>175</th>
      <td>7.759187</td>
      <td>8.967632</td>
      <td>9.382106</td>
      <td>3.951244</td>
      <td>8.341887</td>
      <td>7.436617</td>
    </tr>
    <tr>
      <th>264</th>
      <td>6.978214</td>
      <td>9.177714</td>
      <td>9.645041</td>
      <td>4.110874</td>
      <td>8.696176</td>
      <td>7.142827</td>
    </tr>
    <tr>
      <th>325</th>
      <td>10.395650</td>
      <td>9.728181</td>
      <td>9.519735</td>
      <td>11.016479</td>
      <td>7.148346</td>
      <td>8.632128</td>
    </tr>
    <tr>
      <th>420</th>
      <td>8.402007</td>
      <td>8.569026</td>
      <td>9.490015</td>
      <td>3.218876</td>
      <td>8.827321</td>
      <td>7.239215</td>
    </tr>
    <tr>
      <th>429</th>
      <td>9.060331</td>
      <td>7.467371</td>
      <td>8.183118</td>
      <td>3.850148</td>
      <td>4.430817</td>
      <td>7.824446</td>
    </tr>
    <tr>
      <th>439</th>
      <td>7.932721</td>
      <td>7.437206</td>
      <td>7.828038</td>
      <td>4.174387</td>
      <td>6.167516</td>
      <td>3.951244</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Detergents_Paper':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>161</th>
      <td>9.428190</td>
      <td>6.291569</td>
      <td>5.645447</td>
      <td>6.995766</td>
      <td>1.098612</td>
      <td>7.711101</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Delicatessen':
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.248504</td>
      <td>9.724899</td>
      <td>10.274568</td>
      <td>6.511745</td>
      <td>6.728629</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>137</th>
      <td>8.034955</td>
      <td>8.997147</td>
      <td>9.021840</td>
      <td>6.493754</td>
      <td>6.580639</td>
      <td>3.583519</td>
    </tr>
    <tr>
      <th>142</th>
      <td>10.519646</td>
      <td>8.875147</td>
      <td>9.018332</td>
      <td>8.004700</td>
      <td>2.995732</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>183</th>
      <td>10.514529</td>
      <td>10.690808</td>
      <td>9.911952</td>
      <td>10.505999</td>
      <td>5.476464</td>
      <td>10.777768</td>
    </tr>
    <tr>
      <th>184</th>
      <td>5.789960</td>
      <td>6.822197</td>
      <td>8.457443</td>
      <td>4.304065</td>
      <td>5.811141</td>
      <td>2.397895</td>
    </tr>
    <tr>
      <th>187</th>
      <td>7.798933</td>
      <td>8.987447</td>
      <td>9.192075</td>
      <td>8.743372</td>
      <td>8.148735</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>203</th>
      <td>6.368187</td>
      <td>6.529419</td>
      <td>7.703459</td>
      <td>6.150603</td>
      <td>6.860664</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>233</th>
      <td>6.871091</td>
      <td>8.513988</td>
      <td>8.106515</td>
      <td>6.842683</td>
      <td>6.013715</td>
      <td>1.945910</td>
    </tr>
    <tr>
      <th>285</th>
      <td>10.602965</td>
      <td>6.461468</td>
      <td>8.188689</td>
      <td>6.948897</td>
      <td>6.077642</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>289</th>
      <td>10.663966</td>
      <td>5.655992</td>
      <td>6.154858</td>
      <td>7.235619</td>
      <td>3.465736</td>
      <td>3.091042</td>
    </tr>
    <tr>
      <th>343</th>
      <td>7.431892</td>
      <td>8.848509</td>
      <td>10.177932</td>
      <td>7.283448</td>
      <td>9.646593</td>
      <td>3.610918</td>
    </tr>
  </tbody>
</table>
</div>



```python
## 计算大于等于两次的异常数据

def caltwiceoutliers(row):
    out = 0
    for feature in log_data.keys():
        p = row[feature]
        f = log_data[feature]
        Q1 = np.percentile(f,25)
        Q3 = np.percentile(f,75)
        step = 1.5*(Q3-Q1)
        if (p <= Q1 - step) or (p >= Q3 + step): 
            out += 1
    return out > 1 

lst = []
for i,row in log_data.iterrows():
    if caltwiceoutliers(row): 
        lst.append(i)
display(log_data.loc[lst])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
  </tbody>
</table>
</div>


### 问题 4
* 根据上述定义，有任何数据点属于多个特征的离群值吗？
* 应该从数据集中删除这些数据点吗？
* 如果向 `outliers` 列表中添加了任何要移除的数据点，请解释为何这么做。

** 提示：**如果有数据点在多个类别中都属于离群值，思考下为何是这种情况，以及是否确实需要删除。此外注意离群值对 K 均值有何影响，以及这种影响对分析是否删除这些数据起到决定作用。

**答案：**

** 回答：**
 [65,66,75,128,154]这几个样本在不同的特征上都超出离群值步长范围，表现为异常特征。删除前后对于PCA最大方差的数值并见影响，所以可以不删除。

## 特征转换
在此部分，你将利用主成分分析 (PCA) 得出批发客户数据的基本结构。因为对数据集使用 PCA 会计算哪些维度最适合最大化方差，我们将发现哪些特征组合最能描述客户。

### 实现：PCA

现在数据已经缩放为更正态的分布，并且删除了任何需要删除的离群值，现在可以向 `good_data` 应用 PCA，以发现哪些数据维度最适合最大化所涉及的特征的方差。除了发现这些维度之外，PCA 还将报告每个维度的*可解释方差比*——数据中有多少方差可以仅通过该维度进行解释。注意 PCA 的成分（维度）可以视为空间的新“特征”，但是它是数据中存在的原始特征的成分。

你需要在下面的代码块中实现以下步骤：
 - 导入 `sklearn.decomposition.PCA` 并将对 `good_data` 进行六维度 PCA 转化的结果赋值给 `pca`。
 - 使用 `pca.transform` 对 `log_samples` 应用 PCA 转化，并将结果赋值给 `pca_samples`。


```python
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
good_data = log_data
pca = PCA(n_components=6)
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)
```


    
![png](customer_segments-zh_files/customer_segments-zh_23_0.png)
    


### 问题 5

* 由第一个主成分和第二个主成分解释的数据方差* **总量** *是多少？
* 前四个主成分解释的数据方差是多少？
* 使用上面提供的可视化图表描述每个维度和每个维度解释的累积方程，侧重于每个维度最能表示哪些特征（包括能解释的正方差和负方差）。讨论前四个维度最能表示什么样的客户消费规律。

**提示：**特定维度的正增长对应的是*正加权*特征的_增长_以及*负加权*特征的_降低_。增长或降低比例由具体的特征权重决定。

**答案：**


* 第一个主成分和第二个主成分解释的数据方差总量是：0.719；

* 前四个主成分解释的数据方差是0.9314；

* 维度一特征权重高的：清洁剂，牛奶，杂货，代表客户为零售商店。

* 维度二特征权重高的：新鲜食品，冷冻食品，和熟食 代表客户为餐厅。

* 维度三特征权重高的：新鲜食品、熟食 代表客户为快餐店

* 维度四特征权重高的：冷冻食品，熟食为主 代表客户为西餐厅


### 观察
运行以下代码，看看经过对数转换的样本数据在六维空间里应用 PCA 转换后有何变化。观察样本数据点的前四个维度的数字值。看看与你一开始对样本数据点的判断是否一致。


```python
# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
      <th>Dimension 3</th>
      <th>Dimension 4</th>
      <th>Dimension 5</th>
      <th>Dimension 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.5699</td>
      <td>1.6623</td>
      <td>-2.1600</td>
      <td>-1.3119</td>
      <td>-0.4088</td>
      <td>-0.1069</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.3702</td>
      <td>-1.7971</td>
      <td>0.1871</td>
      <td>0.3020</td>
      <td>-0.5955</td>
      <td>0.0546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.7301</td>
      <td>-0.6900</td>
      <td>-1.3686</td>
      <td>0.8509</td>
      <td>-0.7943</td>
      <td>-0.1793</td>
    </tr>
  </tbody>
</table>
</div>


### 实现：降维
在使用主成分分析时，主要目标之一是降低数据维度，以便降低问题的复杂度。降维有一定的代价：使用的维度越少，则解释的总方差就越少。因此，为了了解有多少个维度对问题来说是必要维度，*累积可解释方差比*显得极为重要。此外，如果大量方差仅通过两个或三个维度进行了解释，则缩减的数据可以之后可视化。

你需要在下面的代码块中实现以下步骤：
 - 将对 `good_data` 进行二维拟合 PCA 转换的结果赋值给 `pca`。
 - 使用 `pca.transform` 对 `good_data` 进行 PCA 转换，并将结果赋值给 `reduced_data`。
 - 使用 `pca.transform` 应用 `log_samples`  PCA 转换，并将结果赋值给 `pca_samples`。


```python
# TODO: Apply PCA by fitting the good data with only two dimensions
pca =  PCA(n_components=2)
pca.fit(good_data)
# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)
# TODO: Transform log_samples using the PCA fit above
pca_samples =  pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
```

### 观察
运行以下代码，看看经过对数转换的样本数据在仅使用二个维度并应用 PCA 转换后有何变化。观察前两个维度的值与六维空间里的 PCA 转换相比如何没有变化。


```python
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.5699</td>
      <td>1.6623</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.3702</td>
      <td>-1.7971</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.7301</td>
      <td>-0.6900</td>
    </tr>
  </tbody>
</table>
</div>


## 可视化双标图
双标图是一种散点图，每个数据点由主成分上的分数表示。坐标轴是主成分（在此图中是 `Dimension 1` 和 `Dimension 2`）。此外，双标图显示了原始特征沿着成分的投影情况。双标图可以帮助我们解释降维数据，并发现主成分与原始特征之间的关系。

运行以下代码单元格，以生成降维数据双标图。


```python
# Create a biplot
vs.biplot(good_data, reduced_data, pca)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23480c59898>




    
![png](customer_segments-zh_files/customer_segments-zh_31_1.png)
    


### 观察

获得原始特征投影（红色部分）后，更容易解释每个点在散点图中的相对位置。例如，图中右下角的点更有可能对应于在 `'Milk'`、`'Grocery'` 和 `'Detergents_Paper'` 上花费很多、但是在其他产品类别上花费不多的客户。

根据该双标图，哪些原始特征与第一个成分的关系最紧密？哪些特征与第二个成分的关系最紧密呢？这些观察结果与你之前获得的 pca_results 图表吻合吗？ 

## 聚类

在此部分，你将选择使用 K 均值聚类算法或高斯混合模型聚类算法发现数据中隐藏的各种客户细分。然后，你将通过将数据点重新转换成原始维度和范围，从聚类中还原具体的数据点以了解它们的显著性。

### 问题 6

* 使用 K 均值聚类算法有何优势？
* 使用高斯混合模型聚类算法有何优势？
* 根据你对批发客户数据到目前为止观察到的结果，你将使用这两个算法中的哪个，为何？

** 提示： **思考下硬聚类和软聚类之间的区别，以及哪种聚类适合我们的数据集。

**答案：**

* k-means 算法优的势：

 * 是解决聚类问题的一种经典算法，简单、快速

 * 对处理大数据集，该算法保持可伸缩性和高效性

 * 聚类是密集的，且类与类之间区别明显时，效果较好。

 
 
* 高斯混合模型聚类算法的优势：

 *  类区别划分不是那么明显时
 
 *  计算快，是学习混合模型最快的算法
 
 * 低偏差，不容易欠拟合
 
由于Kmeans的优势是密集型类与类之间区别明显时效果较好，本数据离散值比较多，而且有很多相同属性，所以采用高斯混合模型聚类算法。
 
### 实现：创建聚类
根据具体的问题，你预计从数据中发现的距离数量可能是已知的数量。如果无法根据*先验*判断聚类的数量，则无法保证给定的聚类数量能够以最佳方式细分数据，因为不清楚数据存在什么样的结构（如果有的话）。但是，我们可以根据每个数据点的*轮廓系数*量化聚类的“优势” 。数据点的[轮廓系数](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)会衡量数据点与所分配的聚类之间的相似度，程度用 -1（不相似）到 1（相似）表示。计算*均值*轮廓系数是对给定聚类进评分的简单方法。

你需要在下面的代码块中实现以下步骤：
 - 对 `reduced_data` 应用聚类算法并将结果赋值给 `clusterer`。
 - 使用 `clusterer.predict` 预测 `reduced_data` 中每个数据点的聚类，并将它们赋值给 `preds`。
 - 使用算法的相应属性得出聚类中心，并将它们赋值给 `centers`。
 - 预测 `pca_samples` 中每个样本数据点的聚类，并将它们赋值给 `sample_preds`。
 - 导入 `sklearn.metrics.silhouette_score` 并对照 `preds`计算 `reduced_data` 的轮廓分数。
   - 将轮廓分数赋值给 `score` 并输出结果。


```python
# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
best_score =0
for i in [2, 3, 4, 5, 6, 7, 8]:
    clusterer = GaussianMixture(n_components=i, random_state=0)
    clusterer.fit(reduced_data)
 
    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
     #边界系数，[-1, 1] 越大越清晰
    score = silhouette_score(reduced_data, preds)
    
    print("For clusters = ", i,
      "The score is :", score)
    if (score > best_score):
        best_clusterer = clusterer
        best_score = score
        best_cluster = i
        
print("For clusters = ",best_clusterer.n_components,'The score is best!')
    

```

    For clusters =  2 The score is : 0.4099683245278784
    For clusters =  3 The score is : 0.40194571937717044
    For clusters =  4 The score is : 0.31214203486720543
    For clusters =  5 The score is : 0.276392991643947
    For clusters =  6 The score is : 0.30088433392758923
    For clusters =  7 The score is : 0.22666071211515948
    For clusters =  8 The score is : 0.26631198668498973
    For clusters =  2 The score is best!
    

### 问题 7

* 报告你尝试的多个聚类数量的轮廓分数。
* 在这些数量中，哪个聚类数量的轮廓分数最高？

**答案：**

* 2个聚类数量的轮廓分数最高

### 聚类可视化
使用上述评分指标为你的聚类算法选择最佳聚类数量后，现在可以通过执行以下代码块可视化结果了。注意，为了进行实验，你可以随意调整你的聚类算法的聚类数量，以查看各种不同的可视化结果。但是，提供的最终可视化图表应该对应的是最佳聚类数量。


```python
# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)
```


    
![png](customer_segments-zh_files/customer_segments-zh_35_0.png)
    


### 实现：数据恢复
上述可视化图表中出现的每个聚类都有一个中心点。这些中心（或均值）并不是来自数据中的特定数据点，而是相应聚类预测的所有数据点的*平均值*。对于创建客户细分这个问题来说，聚类的中心点对应的是*该细分的平均客户数量*。因为数据目前是降维状态并且进行了对数缩放，我们可以通过应用逆转换从这些数据点中还原代表性客户支出。

你需要在下面的代码块中实现以下步骤：
 - 使用 `pca.inverse_transform` 对 `centers` 应用逆转换，并将新的中心点赋值给 `log_centers`。
 - 使用 `np.exp` 对 `log_centers` 应用 `np.log` 的逆函数，并将真正的中心点赋值给 `true_centers`。


```python
# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>9039.0</td>
      <td>2225.0</td>
      <td>2791.0</td>
      <td>2053.0</td>
      <td>378.0</td>
      <td>754.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>2951.0</td>
      <td>7554.0</td>
      <td>12489.0</td>
      <td>784.0</td>
      <td>4671.0</td>
      <td>849.0</td>
    </tr>
  </tbody>
</table>
</div>


### 问题 8

* 思考上述代表性数据点的每个产品类别的总购买成本，并参考该项目开头的数据集统计学描述（具体而言，查看各个特征点的均值）。每个客户细分可以表示什么样的场所集合？

**提示：**分配给 `'Cluster X'`  的客户应该与 `'Segment X'` 的特征集表示的场合最一致。思考每个细分表示所选特征点的什么值。参考这些值并通过均值了解它们表示什么样的场合。

**答案：**
* segment0 Fres、Milk、Grocery、Frozen、Detergents_Paper和Delicatessen所有特征的均少于均值应该是小型零售店；

* segment1 Milk、Detergents_Paper、Grocery 超过均值、其他均低于平均值数，应该代表牛奶咖啡厅；



### 问题 9

* 对于每个样本点，* **问题 8** *中的哪个客户细分最能代表它？
* 每个样本点的预测与此细分保持一致吗？*

运行以下代码块，看看每个样本点预测属于哪个聚类。


```python
# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
```

    Sample point 0 predicted to be in Cluster 1
    Sample point 1 predicted to be in Cluster 1
    Sample point 2 predicted to be in Cluster 0
    


```python
display(pd.concat([true_centers,samples]))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>9039.0</td>
      <td>2225.0</td>
      <td>2791.0</td>
      <td>2053.0</td>
      <td>378.0</td>
      <td>754.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>2951.0</td>
      <td>7554.0</td>
      <td>12489.0</td>
      <td>784.0</td>
      <td>4671.0</td>
      <td>849.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>8590.0</td>
      <td>3045.0</td>
      <td>7854.0</td>
      <td>96.0</td>
      <td>4095.0</td>
      <td>225.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11594.0</td>
      <td>7779.0</td>
      <td>12144.0</td>
      <td>3252.0</td>
      <td>8035.0</td>
      <td>3029.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28257.0</td>
      <td>944.0</td>
      <td>2146.0</td>
      <td>3881.0</td>
      <td>600.0</td>
      <td>270.0</td>
    </tr>
  </tbody>
</table>
</div>


# **答案：**

* 样本0和样本1 在Detergents_papaer,Grocery，Milk，更接近聚类1的中心点,其他可能更聚类1

* 样本2在Fresh，Detergents_Paper，Frozen，Grocery，Milk均接近聚类0的中心点，更可能属于类别0

* 每个样本点的预测与此细分保持一致

## 总结

在最后一部分，你将研究可以对聚类数据采用的方式。首先，你将思考特定的送货方案对不同的客户群（即***客户细分***）有何不同影响。接着，你将思考为每个客户设定标签（该客户属于哪个*细分*）可以如何提供关于客户数据的额外特征。最后，你将比较***客户细分***和数据中的隐藏变量，看看聚类分析是否发现了特定的关系。

### 问题 10
公司在对自己的产品或服务做出小小的改变时，经常会运行 [A/B 测试](https://en.wikipedia.org/wiki/A/B_testing)，判断这项改变对客户有正面还是负面影响。批发商打算将送货服务从目前的一周 5 天改成一周 3 天。但是，批发商仅针对会为其带来正面影响的客户做出这一送货服务变更。 

* 批发商可以如何使用客户细分判断哪些客户（如果有）对送货服务变化保持正面响应。？*

**提示：**可以假设变化会平等地影响到所有客户吗？如何判断对哪些客户群的影响最大？

**答案：**
* 首页我们了解A/B 测试，简单来说，就是为一个目标制定两套方案A和B，让一部分客户使用A 方案，一部分客户用户使用 B 方案，并且A和B方案的客户群里占比一致，并且登记那部分客户影响比较大就是用哪套方案。
* 对于本项目的客户分类超市/零售店/咖啡厅（如果有）等客户群里，根据群体性质的不同，例如咖啡厅对某一产品需求要求比较高。更少的配送周期对于保证商品的新鲜度来说会有更正面的响应。小型超市由于消费量比较大，也会有更好的正面方面，相反零售店的量比较少，商品销售的周期相对会比较厂。正面的影响会相对较弱。所以根据库存销售的周期已经对客户对商品的要求，不同分类的客户对这种改变的正面影响会不一样。

### 问题 11
在使用聚类技巧时，我们从原始无标签数据中得出了额外的结构。因为每个客户都属于某个最合适的***客户细分***（取决于应用的聚类算法），我们可以将*”客户细分“*看作数据的**工程化特征**。假设批发商最近吸引了 10 个新的客户，每个客户都能为每个产品类别带来预期的年收入（估值）。了解这些估值后，批发商希望将每个新客户归类到一个***客户细分***，以确定最合适的送货服务。 
* 批发商如何仅使用估计的产品开支和**客户细分**数据为新客户设定标签？

**提示：**可以使用监督式学习器对原始客户进行训练。目标变量可以是什么？

**答案：**

可以根据目前6个特征的中心点和每个特征的中位数，针对新引进的10位新客户，根据他们每个产品类别的估值更哪几种类别更接近某一种来设置客户标签。

### 可视化底层分布图

在该项目开始时，我们提到我们会从数据集中排除 `'Channel'` 和 `'Region'` 特征，以便在分析过程中侧重于客户产品类别。通过向数据集中重新引入 `'Channel'` 特征，在考虑之前对原始数据集应用的相同 PCA 降维算法时，发现了有趣的结构。

运行以下代码块，看看每个数据点在降维空间里为何标记成 `'HoReCa'`（酒店/餐厅/咖啡厅）或 `'Retail'`。此外，你将发现样本数据点在图中被圈起来了，这样可以标识它们的标签。


```python
# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)

```


    
![png](customer_segments-zh_files/customer_segments-zh_42_0.png)
    


### 问题 12

* 你所选的聚类算法和聚类数量与此酒店/餐厅/咖啡厅客户到零售客户底层分布图相比，效果如何？
* 根据此分布图，有任何客户细分可以完全分类为”零售“或”酒店/餐厅/咖啡厅“客户吗？
* 你认为这些分类与之前的客户细分定义保持一致吗？

**答案：**
我选择的聚类算法和聚类数量，与与酒店/餐厅/咖啡厅客户和零售客户分布图相比，效果还不错，但也还存在一些异常数据。根据Dimension1以看到划分'零售客户'或者是'酒店/餐厅/咖啡厅客户'分布还不错。这些分类和之前的客户细分定义大体一致。但选择的样本数据点属于异常数据，咖啡厅购买的商品类别和其他餐馆不一致，而零售商由于没有特别类别的需求和购买量，基本上各种商品都比较少且杂。因此会处于边界周边。
> **注意**：完成所有代码实现部分并成功地回答了上述每个问题后，你可以将该 iPython Notebook 导出为 HTML 文档并获得最终要提交的项目。为此，你可以使用上面的菜单或依次转到
> *文件 -> 下载为 -> HTML (.html)**。在提交时，请同时包含该 notebook 和完成的文档。


```python

```


```python

```
