from sklearn.datasets import load_diabetes#dataset:数据集模块
from sklearn.preprocessing import StandardScaler#preprocessing:数据预处理模块，StandardScaler:z-score标准化
from sklearn.model_selection import train_test_split#model_selection:数据划分模块，train_test_split:自动拆分训练集、测试集函数
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error#metrics:模型评估指标模块，mean_squared_error=MSE均方误差(回归损失函数),r2_score:R方决定系数,衡量模型对数据变异的解释程度（越接近1越好）,mean_absolute_error=MAE平均绝对误差
import numpy as np
import matplotlib.pyplot as plt

#使用sklearn 糖尿病数据集
data=load_diabetes()
X,y=data.data,data.target#X特征矩阵，y目标值

#标准化
scaler=StandardScaler()#scaler:标准化容器
X_scaled=scaler.fit_transform(X)#fit()计算均值、标准差，transform()用均值标准差标准化数据

#划分数据集
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)#random_state=42,固定随机种子，保证每次划分结果一样


### 解析解实现 ###
#添加偏置项
X_b=np.c_[np.ones((X_train.shape[0],1)),X_train]
#np.ones():创建全1数组,X_train.shape[0]:shape尺寸,[0]是行数(取输出结果的第一个)，np.c_[]按列拼接数组
#给公式：预测值=θ₀+θ₁×特征1+θ₂×特征2+...+θ₁₀×特征10 加上前面的θ₀偏置项1,θ₀*1,除了原有的n列特征，左边还多了一列全是1的特征

#计算闭式解
theta_best=np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
#T:矩阵转置，dot()矩阵点乘

#预测函数
def predict(X):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return X_b.dot(theta_best)


### 梯度下降实现 ###
def gradient_descent(X,y,learning_rate=0.01,n_iters=1000):#学习率决定每次参数更新幅度，n_iters是迭代次数
    m=X.shape[0]#获取样本总个数
    X_b=np.c_[np.ones((m,1)),X]
    theta=np.random.randn(X_b.shape[1],1)#随机生成一组初始的参数θ
    loss_history=[]#记录每次迭代的损失值

    for _ in range(n_iters):
        loss=np.sum((X_b.dot(theta)-y.reshape(-1,1))**2)/ (2 * m)
        loss_history.append(loss)

        gradients=2/m*X_b.T.dot(X_b.dot(theta)-y.reshape(-1,1))#目标函数J(θ)对θ求偏导
        theta-=learning_rate*gradients
    return theta,loss_history

#针对梯度下降的预测函数
def predict_gd(X,theta):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return X_b.dot(theta)

#比较不同学习率对梯度下降的影响,绘制损失函数下降曲线
learning_rates=[0.001,0.01,0.1]
gd_thetas={}

plt.figure(figsize=(10,6))
for lr in learning_rates:
    theta_gd,loss_hist=gradient_descent(X_train,y_train,learning_rate=lr,n_iters=1000)
    gd_thetas[lr]=theta_gd
    plt.plot(loss_hist,label=f'LR={lr}')

plt.xlabel('Iterations')
plt.ylabel('Loss $J(\\theta)$')
plt.title('Loss Function Descent Curve (Different Learning Rates)')
plt.legend()
plt.grid(True)
plt.show()

#模型评估
#解析解评估
print("解析解模型评估")
y_pred=predict(X_test)
print(f"MSE:{mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2:{r2_score(y_test,y_pred):.2f}")
print(f"MAE:{mean_absolute_error(y_test,y_pred):.2f}")
print("梯度下降模型评估")
y_pred_gd = predict_gd(X_test, gd_thetas[0.1])
print(f"MSE: {mean_squared_error(y_test, y_pred_gd):.2f}")
print(f"R^2: {r2_score(y_test, y_pred_gd):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_gd):.2f}")

#不同特征组合对结果的影响
X_train_sub=X_train[:,:3]
X_test_sub=X_test[:,:3]

X_b_sub=np.c_[np.ones((X_train_sub.shape[0],1)),X_train_sub]
theta_best_sub=np.linalg.pinv(X_b_sub.T.dot(X_b_sub)).dot(X_b_sub.T).dot(y_train)

X_test_b_sub=np.c_[np.ones((X_test_sub.shape[0],1)),X_test_sub]
y_pred_sub=X_test_b_sub.dot(theta_best_sub)

print("仅使用前3个特征的解析解模型评估")
print(f"MSE: {mean_squared_error(y_test, y_pred_sub):.2f}")
print(f"R^2: {r2_score(y_test, y_pred_sub):.2f}")

#可视化结果
plt.scatter(y_test, y_pred, label='Analytical Solution')
# 绘制梯度下降散点，对比差异
plt.scatter(y_test, y_pred_gd, marker='x', label='Gradient Descent')
# 绘制理想拟合线，并添加标签对比差异
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()