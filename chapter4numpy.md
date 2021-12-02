# Numpy

numpy的功能如下：

1. ndarray，多维数组，具有矢量算术运算和复杂广播能力
2. 对整组数据进行快速运算的标准数学函数（不需要写循环）
3. 读写磁盘数据的工具和操作内存映射文件的工具
4. 线性代数，随机数生成以及傅里叶变换
5. 集成由CC++等语言编写代码的工具（numpy提供了一个简单的C的API，很容易将数据传递给由低级语言编写的外部库，外部库也可以返回数据给python）


大部分数据分析应用关注的功能集中在：
1. 数据整体和清洁，子集构造和过滤，转换等
2. 常用的数组算法，排序，唯一化和集合运算


## Numpy的ndarray：一种多维数据对象

ndarray是一个通用的同构数据多维容器，其中的所有元素属于同一类型。
每个数组有shape和dtype
shape表示各维度大小的元组
dtype说明数组数据的类型

## 创建ndarray

1. 使用array函数：接收一切序列型的对象，可以是其他数组，然后产生一个新的含有传入数据的numpy数组

```python
#列表转换
import numpy as np
arr1=np.array(data1)
arr1
Out[13]: array([6. , 7.5, 8. , 0. , 1. ])
#嵌套序列（比如一组等长列表组成的列表）
data2=[[1,2,3,4],[5,6,7,8]]
arr2=np.array(data2)
arr2

Out[16]: 
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

arr2.ndim
Out[17]: 2
arr2.shape
Out[18]: (2, 4)
#可以明显的看出，arr1将元素的类型都转换成了浮点数
arr1.dtype
Out[19]: dtype('float64')
```

2. zeros和ones和empty也可以创建数组
   1. zeros和ones分别指定长度或者形状为全0或全1的数组
   2. empty可以创建一个没有具体值的数组
利用上述方法创建的数组需要指定数组的形状，参数是一个表示形状的元组(a,b) a表示行标 b表示列标


*如果没有特别指定，数据类型基本都是float64*

## ndarray的数据类型

dtype是一个特殊的对象，利用np.dtype 可以输出数组元素的数据类型，在定义数组时也可以对dtype进行绑定，将元素指定为特定的数据类型

```python
arr1=np.array([1,2,3],dtype=np.float64)
arr2=np.array([1,2,3],dtype=np.int32)
arr1.dtype

Out[26]: dtype('float64')

# 利用 astype 显式转换其dtype

#astype会创建出一个新的数组，dtype只是数组的一个对象。可以对其进行绑定


arr=np.array([1,2,3,4,5])
arr.dtype
Out[28]: dtype('int32')
float_arr=arr.astype(np.float64)
float_arr.dtype
Out[30]: dtype('float64')



```

## 数组和标量之间的运算

数组间的运算 arr*arr arr-arr
数组与标量的运算 1/arr arr*0.5 将运算应用到元素级


不同大小的数组之间的运算叫做广播

## 基本的索引和切片

```python
#利用arange函数创建等差数组，也可以设置起点终点步长
arr=np.arange(10)
arr
Out[49]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[5]
Out[50]: 5
arr[5:8]
Out[51]: array([5, 6, 7])
arr[5:8]=12
arr
Out[53]: array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])
#需要注意的是数组切片是原始数组，不会复制

#二维数组，索引位置上的元素不再是标量而是一维数组
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
Out[55]: 
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
# arr[2]索引了第三行的所有元素
arr[2]
Out[56]: array([7, 8, 9])


# 三维数组索引，返回的就是维度低一点的数组,但是含有所有的数据
arr3d=np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]])
arr3d
Out[60]: 
array([[[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]]])
arr3d[0]
Out[61]: 
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])

# arr3d[1,0]可以访问索引以(1,0)开头的值，返回一维数组



```


### 切片索引

```python
arr2d
Out[85]: 
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
arr2d[:2]
Out[86]: 
array([[1, 2, 3],
       [4, 5, 6]])
arr2d[:2,1:]
Out[87]: 
array([[2, 3],
       [5, 6]])

```

### 布尔型索引

现有两个数组，一个用于存储数据以及一个存储姓名

``` python

names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data=randn(7,4)
names
Out[91]: array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')
data
Out[92]: 
array([[-1.5558233 , -0.07617629,  0.04973506, -0.01015275],
       [ 0.04535325, -0.2324176 ,  1.33135032,  0.14820336],
       [-0.18234659,  1.22571665, -0.27913231, -0.88359976],
       [ 0.4681641 , -0.95247673, -1.11766119, -0.01085024],
       [ 1.67682234,  0.10432897,  0.53481638,  0.42695197],
       [-2.18481876,  1.14532299,  1.11959121,  0.24586887],
       [-1.73779574, -1.03856874,  1.09938498, -0.3426236 ]])

#假设每个名字对应了数组的一行数据，现在我们要找出名字是'Bob'所对应的行
#先对names进行一个比较运算，该运算也是矢量化的
names=='Bob'
#names每个元素与Bob进行运算产生一个布尔型数组
Out[93]: array([ True, False, False,  True, False, False, False])
#利用布尔数组用于数组索引
data[names=='Bob']
# 输出数据数组的第0行和第3行
Out[94]: 
array([[-1.5558233 , -0.07617629,  0.04973506, -0.01015275],
       [ 0.4681641 , -0.95247673, -1.11766119, -0.01085024]])
# 布尔型数长度必须跟被索引的轴长度一致，将布尔型数组跟切片整数混合使用
#python关键字and or在布尔型数组中无效
mask=((names=='Bob' )| (names== 'Will'))  #比较对象之间不加括号报错
#利用布尔型数组对数组值进行设置
#将data中小于0的值设为0
data[data<0]=0
```

### 花式索引

## 数组转置和轴对称
arr=np.arange(15).reshape((3,5))
arr
Out[126]: 
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
arr.T
Out[127]: 
array([[ 0,  5, 10],
       [ 1,  6, 11],
       [ 2,  7, 12],
       [ 3,  8, 13],
       [ 4,  9, 14]])


## 通用函数：快速的元素级数组函数
通用函数(ufunc)是一种对ndarray中的数据执行元素级运算的函数。

例如 np.sqrt(arr);np.exp(arr),这些都是一元ufunc
另外有一些(add，maximum)接收两个数组，叫做二元，返回一个结果数组


## 利用数组进行数据处理

Numpy数组可以使得许多数据处理任务表述为简单的数组表达式，用数组表达式代替循环的做法，叫做矢量化。


```python
#np.meshgird()接收两个一维数组并产生两个二维数组。


points=np.arange(-5,5,0.01) #arange()指定了起始以及终点位置，并规定了步长。
xs,ys=np.meshgrid(points,points)


```

## 将条件逻辑表述为数组运算

numpy.where 函数是三元表达式 x if condition else y 的矢量化版本。
``` pyhton

xarr=np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,False])
result=[(x if c else y) for x,y,c in zip(xarr,yarr,cond)]
result
Out[159]: [1.1, 2.2, 1.3, 2.4]

```

## 数学和统计方法
可以通过数组上的一组数学函数对*整个数组*或*某个轴向*的数据进行统计计算。
sum mean 以及标准差std等聚合计算。既可以当作数组的实例对象使用，也可以当作顶级numpy函数使用

mean 和sum函数可以接受一个 axis参数，最终结果是一个少一维的数组


其他诸如cumsum和cumprod的方法则不聚合，而是产生一个由中间结果组成的数组。


arr=np.array([[0,1,2],[3,4,5],[6,7,8]])

cunsum 本身表示对指定轴数据求和，axis=0表示对列求和， axis=1表示对行求和。
arr.cumsum(0)
arr.cumsum(1)

对指定轴的数据求积
arr.cumprod(1)

## 用于布尔型数组的方法


布尔值数组在上面的运算中会进行转换，True转换为1 false则转换为0

arr=randn(100)
arr2=(arr>0)
得到一个布尔型数组，大于0的元素就是True 小于0的元素就是false

另外有两个方法any和all，对布尔型数组非常有用。
any用于测试数组中是否存在一个或多个True，而all则检查数组中所有的值是否都是True

## 排序

Numpy数组可以通过sort*就地排序* & *顶级排序*：np.sort()


*就地排序*会修改数组本身，*顶级方法*则返回数组的已排序对象。
```python
arr=randn(8)
arr.sort()


arr=randn(5,3)
#sort()函数可设置的参数 轴向，轴向为0代表列，轴向为1代表行
arr.sort(1)




```

## 唯一化以及其他的集合逻辑
Numpy提供了一些针对一维ndarray的基本集合运算。 最常见的是 np.unique,找出数组中*唯一值*并返回*已经排序*的结果

```python
names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
np.unique(names)


Out[199]: array(['Bob', 'Joe', 'Will'], dtype='<U4')



ints=np.array([3,3,3,2,2,1,1,4,4])
np.unique(ints)


Out[201]: array([1, 2, 3, 4])



# np.in1d 用来测试一个数组中的值在另一个数组的成员资格，返回一个布尔型数组

values=np.array([6,0,0,3,2,5,6])
# 接收两个数组，判断后一个数组在第一个数组的成员资格，并且返回的布尔型数组是以第一个数组为模板进行创建的。
np.in1d(values,[2,3,6])

Out[203]: array([ True, False, False,  True,  True, False,  True])

```

* unique(x)  计算x中的唯一元素，并返回有序结果。
* intersect1d(x,y) 计算x和y的公共元素，并返回有序结果。
* union1d(x,y)
* in1d(x,y)
* setdifff1d(x,y)
* setxor1d(x,y)  存在于一个数组但不同时存在与两个数组中。



## 用于数组的文件输入输出
读写磁盘上的文本数据或二进制数据

### 将数组以二进制格式保存到磁盘

主要利用*np.save* 和*np.load* 
数组默认以未压缩的原始二进制格式保存在 .npy文件中
```python
arr=np.arange(10)
np.save('some_array',arr)
np.load('some_array.npy')
Out[206]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#通过np.savez 可以将多个数组存到一个压缩文件中，将数组以关键字参数的形式传入
np.savez('array_archive.npz',a=arr,b=arr)  # 前面时压缩文件的名字已经格式，后面的关键字参数可以传递多个数组
#加载还是采用load方法。
arch=np.load('array_archive.npz')
#load得到的一个类似字典的对象，通过对应的键去访问相应的数组
arch['b']
```


### 存取文本文件

从文件中加载文本，可以利用pandas的read——csv 和read——table 函数。
有时采用np.loadtxt 或者专门化的 np.genfromtxt 将数据加载到普通的Numpy数组。

## 线性代数

线性代数(矩阵乘法，矩阵分解，行列式以及其他方阵数学)，numpy提供了用于矩阵乘法的 dot函数

``` python
x=np.array([[1.,2.,3.],[4.,5.,6.]])
x.dot(y)  or np.dot(x,y)

Out[215]: 
array([[ 28.,  64.],
       [ 67., 181.]])

# numpy.linalg 中有一组标准的矩阵分解运算以及诸如求逆和行列式之类的东西。   
# 参考P110列出的一些常用numpy.linalg函数
```

## 随机数生成

*常用函数参考P111函数表*