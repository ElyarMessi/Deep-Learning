#!/usr/bin/env python
# coding: utf-8

#手动实现简单的逻辑回归模型
#author:Elyar

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset


# In[2]:


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()


# In[7]:


index=25
plt.imshow(train_set_x_orig[index])


# In[9]:


print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")


# In[11]:


m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
train_set_x_orig.shape


# In[12]:


print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))


# In[13]:


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))


# In[14]:


# Normalization
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# In[15]:


def sigmoid(z):
    return 1/(1+np.exp(-z))

#测试sigmoid()
print("====================测试sigmoid====================")
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(9.2) = " + str(sigmoid(9.2)))


# In[16]:


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0
    assert(w.shape == (dim, 1)) #w的维度是(dim,1)
    assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int
    return (w,b)


# In[21]:


def propagate(w,b,X,Y):
    """
    实现前向和后向传播的成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    
    m = X.shape[1]
    
    #正向传播
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A) )
    
    #计算偏导数
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)
    
    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    # cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {
        "dw":dw,
        "db":db
    }
    
    return (grads,cost)


# In[22]:


#测试一下propagate
print("====================测试propagate====================")
#初始化一些参数
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


# In[23]:


def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    """
    此函数通过运行梯度下降算法来优化w和b
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值
    
    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        costs - 优化期间计算的所有成本列表，将用于绘制学习曲线。
    
    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads['dw']
        db = grads['db']
        
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i,cost))
        
    params = {
        "w":w,
        "b":b
    }
    
    grads = {
        "dw":dw,
        "db":db
    }
    
    return(params,grads,costs)


# In[24]:


#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=300 , learning_rate = 0.01 , print_cost = True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


# In[25]:


def predict(w,b,X):
    """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据
    
    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
    
    """
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape((X.shape[0],1))
    
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i]>0.5 else 0
        
    #使用断言
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction


# In[26]:


#测试predict
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))


# In[43]:


def model_elyar_regression(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=True):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    
    返回：
        model_info  - 包含有关模型信息的字典。
    """
    
    w,b = initialize_with_zeros(X_train.shape[0])
    step = 50
    for i in range(num_iterations//step):
        params,grads,costs = optimize(w,b,X_train,Y_train,step,learning_rate=learning_rate,print_cost=False)
    
        # 此时已经训练完毕
        w,b = params['w'],params['b']
    
        Y_prediction_train = predict(w,b,X_train)
        Y_prediction_test  = predict(w,b,X_test)
    
        #打印训练后的准确性
        print("经过 ",(i+1)*step," 次迭代后：")
        print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
        print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")
    
    model_info = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return model_info


# In[46]:


print("====================测试model====================")     
#这里加载的是真实的数据，请参见上面的代码部分。
d = model_elyar_regression(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1000, learning_rate = 0.0045, print_cost = True)


# In[31]:


costs = d['costs']
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

