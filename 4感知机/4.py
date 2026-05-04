#数据准备：使用线性可分数据集
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

#X:特征矩阵，y:标签向量
X, y = make_classification(n_samples=100, 
                           n_features=2, 
                           n_redundant=0,
                           n_clusters_per_class=1, 
                           class_sep=0.5,
                           flip_y=0,
                           random_state=42)
#n_redundant冗余特征为0，n_clusters_per_class每个类别对应的簇数量为1：数据分布较简单，random_state随机种子，确保每次运行生成的数据完全一致便于复现
y = np.where(y == 0, -1, 1) # 标签映射为{-1, 1}，np.where(condition,x,y)满足条件替换为x，否则y

#数据预处理：标准化并划分训练集/测试集
scaler = StandardScaler()
#实例化标准化，将数据按列（特征）减去均值并除以标准差，使其符合正态分布
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

#随机梯度下降SGD
class Perceptron: 
    def __init__(self, lr=0.01, max_iter=1000): #max_iter最大迭代轮数，防止模型在不可分数据集上死循环
        self.lr = lr 
        self.max_iter = max_iter 
    def fit(self, X, y): 
        self.w = np.zeros(X.shape[1]) #初始化权重向量w为0，shape[0]:行，样本数量，shape[1]:列，特征数量
        self.b = 0 
        self.losses = [] #每一轮迭代中，误分类点产生的总损失
        self.counts = [] #每一轮迭代中，发生误分类的总次数
        for _ in range(self.max_iter): #_表示循环变量在代码中不被使用
            loss = 0.0 
            count = 0
            for xi, yi in zip(X, y): 
                if yi * (np.dot(self.w, xi) + self.b) <= 0: #判断是否被误分类
                    loss += -yi * (np.dot(self.w,xi) + self.b)
                    self.w += self.lr * yi * xi #新参=旧参-学习率*梯度
                    self.b += self.lr * yi 
                    count += 1
                #累加当前误分类点到超平面的距离，舍弃了分母的w（只关注超平面能否把正负例分开，不关心法向量大小），而误分类点的yi(w.xi+b)为负数，再加上-
            self.losses.append(loss) 
            self.counts.append(count)

            if count == 0: 
                break          
        return self
    def predict(self,X):
        return np.where(np.dot(X,self.w) + self.b >= 0,1,-1)
    
#对偶形式
class DualPerceptron: 
    def __init__(self, lr=0.01, max_iter=1000): 
        self.lr = lr 
        self.max_iter = max_iter 
        
    def fit(self, X, y): 
        n_samples, n_features = X.shape 
        # w = w0+n*lr*x1+3n*lr*x2+... = 0+α*x1+α*x2+...
        self.alpha = np.zeros(n_samples) 
        self.b = 0 
        self.gram = X.dot(X.T) # Gram 矩阵
        #Gram[i]表示[xi*xj]一维矩阵，(j=1~N),Gram[i][j]=xi*xj数值
        self.counts=[]
        self.X_fit = X
        self.y_fit = y
        for _ in range(self.max_iter): 
            count=0
            for i in range(n_samples): 
                if y[i] * (np.sum(self.alpha * y * self.gram[i]) + self.b) <= 0: 
                #对偶形式的误分判定逻辑。self.gram[i] 取出第 i 个样本与所有样本的内积 
                    self.alpha[i] += self.lr # αi+=lr
                    self.b += self.lr * y[i]# b+=lr*yi
                    count+=1
            self.counts.append(count)
            if count == 0:
                break
        return self
    #由于对偶形式只学习出了α，预测时需要借助训练数据还原权重 w
    def predict(self, X_test):
        w = np.sum((self.alpha * self.y_fit)[:, np.newaxis] * self.X_fit, axis=0)
        return np.where(np.dot(X_test, w) + self.b >= 0, 1, -1)
    

print("\n--- 原始形式与对偶形式效率对比 ---")
# 测试原始形式
start_time = time.time()
p_primal = Perceptron(lr=0.01, max_iter=1000)
p_primal.fit(X_train, y_train)
primal_time = time.time() - start_time
print(f"原始形式 (SGD) 训练耗时: {primal_time:.5f} 秒，迭代轮数: {len(p_primal.losses)}")

# 测试对偶形式
start_time = time.time()
p_dual = DualPerceptron(lr=0.01, max_iter=1000)
p_dual.fit(X_train, y_train)
dual_time = time.time() - start_time
print(f"对偶形式 (Dual) 训练耗时: {dual_time:.5f} 秒，迭代轮数: {len(p_dual.counts)}")


# 批量梯度下降 (BGD)
# 特点：每次参数更新都会使用完整的训练集计算梯度
# 优点：朝着全局最优解或局部最优解的方向稳定下降，更新平滑
# 缺点：当数据集非常大时，计算一次梯度的开销极大，内存占用高
class BGDPerceptron:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
        for _ in range(self.max_iter):
            loss = 0.0
            count = 0

            #BGD需要遍历完所有样本后才更新参数，预备好用于累加梯度的变量
            grad_w = np.zeros(X.shape[1])
            grad_b = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    loss += -yi * (np.dot(self.w, xi) + self.b)
                    # 累加所有误分类点的梯度，先不更新
                    grad_w += yi * xi
                    grad_b += yi
                    count += 1
            # 遍历完所有样本后，利用累加的整体梯度，统一更新一次参数
            self.w += self.lr * grad_w
            self.b += self.lr * grad_b
            self.losses.append(loss)
            if count == 0:
                break
        return self


# 小批量梯度下降 (Mini-batch GD)
# 特点：每次参数更新使用数据集中的一小批（batch）样本计算梯度。
# 优点：结合了 SGD 的计算高效性和 BGD 的收敛稳定性，是现代深度学习最常用的策略。
class MiniBatchPerceptron:
    def __init__(self, lr=0.01, batch_size=16, max_iter=1000, random_state=42):
        self.lr = lr    
        self.batch_size = batch_size # 批大小：每次计算梯度使用的样本数量
        self.max_iter = max_iter
        self.random_state = random_state
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
        n_samples = X.shape[0]
        np.random.seed(self.random_state)
        
        for _ in range(self.max_iter):
            loss = 0.0
            count = 0
            # 每一轮迭代开始前，随机打乱数据集
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 步长为 batch_size，每次取出一小块数据进行梯度计算
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                grad_w = np.zeros(X.shape[1])
                grad_b = 0
                for xi, yi in zip(X_batch, y_batch):
                    if yi * (np.dot(self.w, xi) + self.b) <= 0:
                        loss += -yi * (np.dot(self.w, xi) + self.b)
                        grad_w += yi * xi
                        grad_b += yi
                        count += 1
                # 遍历完当前小批次后，立即使用该批次的累加梯度更新参数
                self.w += self.lr * grad_w
                self.b += self.lr * grad_b
                
            self.losses.append(loss)
            if count == 0:
                break
        return self


# 扩展任务1：自适应学习率感知机
class AdaptivePerceptron:
    def __init__(self, lr0=0.1, decay_rate=0.01, max_iter=1000):
        self.lr0 = lr0                # 初始学习率
        self.decay_rate = decay_rate  # 衰减率
        self.max_iter = max_iter
        
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
        
        step = 0 # 全局更新步数 t
        for _ in range(self.max_iter):
            loss = 0.0
            count = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    # 根据全局更新步数计算当前的自适应学习率
                    current_lr = self.lr0 / (1 + self.decay_rate * step)
                    
                    loss += -yi * (np.dot(self.w, xi) + self.b)
                    self.w += current_lr * yi * xi
                    self.b += current_lr * yi
                    count += 1
                    step += 1 # 每发生一次误分类更新，步数加1
            self.losses.append(loss)
            if count == 0:
                break
        return self


# 扩展任务2：多分类感知机 (One-vs-All 策略)
class MultiClassPerceptronOVA:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.classifiers = [] # 用于装载 K 个二分类感知机
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y) # 获取所有不重复的类别标签
        
        # 针对每一个类别，单独训练一个二分类模型
        for c in self.classes:
            # 制作二分类标签：如果是当前类别，设为 1；其他所有类别设为 -1
            y_binary = np.where(y == c, 1, -1)
            
            model = Perceptron(lr=self.lr, max_iter=self.max_iter)
            model.fit(X, y_binary)
            self.classifiers.append(model)
        return self
        
    def predict(self, X):
        # 初始化一个矩阵，用来存放 K 个模型的打分结果
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i, model in enumerate(self.classifiers):
            # 记录第 i 个模型对所有样本的打分：w·x + b
            scores[:, i] = np.dot(X, model.w) + model.b
        # 找出每个样本得分最高的分类器索引，映射回对应的真实类别
        best_class_indices = np.argmax(scores, axis=1)
        return self.classes[best_class_indices]


#比较不同学习率对收敛的影响
learning_rates = [0.001, 0.01, 0.1]
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    # 实例化并训练模型
    model = Perceptron(lr=lr, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 绘制当前学习率下的损失曲线
    plt.plot(range(len(model.losses)), model.losses, label=f'lr = {lr}', marker='o', markersize=4)

plt.xlabel('Epoch (迭代轮数)')
plt.ylabel('Empirical Risk (经验损失)')
plt.title('Training Loss Curve with Different Learning Rates')
plt.legend() # 显示图例
plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线以便于观察
plt.show()


#实例化并训练模型
perceptron = Perceptron(lr=0.01,max_iter=1000)
perceptron.fit(X_train,y_train)


#损失曲线
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)#划分 1 行 2 列的画板位置，第 1 个子图
plt.plot(range(len(perceptron.losses)), perceptron.losses)
plt.xlabel('Epoch')
plt.ylabel('Empirical Risk (Loss)')
plt.title('Training Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(range(len(perceptron.counts)), perceptron.counts, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassifications')  
plt.title('Misclassification Count Curve')

plt.tight_layout()#自动调整子图参数，使之不重叠
plt.show()

#超平面动态变化
def plot_decision_boundary(model, X, y): 
    #获取输入数据第1个特征向量的最小值和最大值，并向外各延伸1个单位确定横轴范围 。
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1 
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1 
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)) 
    #在横纵轴范围内，各自等间距提取100个点，并通过 meshgrid 交叉生成 100*100 = 10000 个背景网格点

    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]) 
    Z = Z.reshape(xx1.shape) #将预测结果还原为 100*100 的矩阵形态
    
    plt.contourf(xx1, xx2, Z, alpha=0.3) 
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k') 
    plt.xlabel('Feature 1') 
    plt.ylabel('Feature 2') 
    plt.title('Decision Boundary')
    plt.show()
plot_decision_boundary(perceptron, X_train, y_train)


print("\nSGD、BGD 与 Mini-batch GD 运行结果对比")
# 1. 随机梯度下降 (SGD)
start_time = time.time()
sgd_model = Perceptron(lr=0.01, max_iter=1000)
sgd_model.fit(X_train, y_train)
print(f"1. SGD 训练耗时: {time.time() - start_time:.5f} 秒，收敛需要迭代轮数: {len(sgd_model.losses)}")

# 2. 批量梯度下降 (BGD)
start_time = time.time()
bgd_model = BGDPerceptron(lr=0.01, max_iter=1000)
bgd_model.fit(X_train, y_train)
print(f"2. BGD 训练耗时: {time.time() - start_time:.5f} 秒，收敛需要迭代轮数: {len(bgd_model.losses)}")

# 3. 小批量梯度下降 (Mini-batch GD)
start_time = time.time()
mbgd_model = MiniBatchPerceptron(lr=0.01, batch_size=16, max_iter=1000)
mbgd_model.fit(X_train, y_train)
print(f"3. Mini-batch 训练耗时: {time.time() - start_time:.5f} 秒，收敛需要迭代轮数: {len(mbgd_model.losses)}")

# 绘制三种梯度下降方式的损失曲线对比图
plt.figure(figsize=(10, 6))
# 取前50个 Epoch 放大看细节
plt.plot(range(len(sgd_model.losses)), sgd_model.losses, label='SGD (每次1个样本)', alpha=0.8, marker='o', markersize=3)
plt.plot(range(len(bgd_model.losses)), bgd_model.losses, label='BGD (每次全量样本)', alpha=0.8, marker='s', markersize=3)
plt.plot(range(len(mbgd_model.losses)), mbgd_model.losses, label='Mini-batch GD (batch=16)', alpha=0.8, marker='^', markersize=3)

plt.xlabel('Epoch (迭代轮数)')
plt.ylabel('Empirical Risk (总损失)')
plt.title('SGD vs BGD vs Mini-batch GD 损失下降对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


print("\n扩展任务1：自适应学习率感知机测试")
# 实例化自适应学习率模型
adaptive_model = AdaptivePerceptron(lr0=0.1, decay_rate=0.01, max_iter=1000)
adaptive_model.fit(X_train, y_train)

# 实例化一个固定学习率模型作为对比 
fixed_model = Perceptron(lr=0.1, max_iter=1000)
fixed_model.fit(X_train, y_train)

# 绘制对比曲线
plt.figure(figsize=(8, 5))
plt.plot(range(len(adaptive_model.losses)), adaptive_model.losses, 
         label='自适应学习率 (Adaptive LR)', marker='o', markersize=4)
plt.plot(range(len(fixed_model.losses)), fixed_model.losses, 
         label='固定学习率 (Fixed LR=0.1)', marker='x', markersize=4, linestyle='--')

plt.xlabel('Epoch (迭代轮数)')
plt.ylabel('Empirical Risk (经验损失)')
plt.title('自适应学习率 vs 固定学习率 损失下降对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"自适应学习率收敛所需轮数: {len(adaptive_model.losses)}")
print(f"固定学习率收敛所需轮数: {len(fixed_model.losses)}")


print("\n扩展任务2：多分类感知机 (OVA) 测试")
# 生成一个3分类的数据集
X_multi, y_multi = make_classification(n_samples=150, n_features=2, n_redundant=0, 
                                       n_informative=2, n_clusters_per_class=1, 
                                       n_classes=3, class_sep=1.5, random_state=42)

# 数据标准化
X_multi_scaled = scaler.fit_transform(X_multi)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi_scaled, y_multi, test_size=0.2, random_state=42)

# 实例化并训练 OVA 多分类模型
ova_model = MultiClassPerceptronOVA(lr=0.01, max_iter=1000)
ova_model.fit(X_train_m, y_train_m)

# 预测并计算准确率
y_pred_m = ova_model.predict(X_test_m)
accuracy = np.mean(y_pred_m == y_test_m)
print(f"多分类 OVA 模型在测试集上的准确率: {accuracy * 100:.2f}%")

# 针对多分类重新写一个画图函数
def plot_multiclass_decision_boundary(model, X, y):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), 
                           np.linspace(x2_min, x2_max, 200))
    
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    plt.figure(figsize=(8, 6))
    # 使用 Paired 颜色映射来区分3个类别
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'多分类决策边界 (One-vs-All)\n准确率: {accuracy*100:.1f}%')
    plt.show()

# 绘制多分类结果
plot_multiclass_decision_boundary(ova_model, X_train_m, y_train_m)


print("\n噪声数据测试 (flip_y=0.3)")
X_noise, y_noise = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                       n_clusters_per_class=1, class_sep=0.5, 
                                       flip_y=0.3, random_state=42)
y_noise = np.where(y_noise == 0, -1, 1)
X_noise_scaled = scaler.fit_transform(X_noise)

p_noise = Perceptron(lr=0.01, max_iter=200) # 减少max_iter防止死循环太久
p_noise.fit(X_noise_scaled, y_noise)

plt.figure(figsize=(6, 4))
plt.plot(range(len(p_noise.losses)), p_noise.losses, color='red')
plt.xlabel('Epoch')
plt.ylabel('Empirical Risk')
plt.title('噪声数据下的损失曲线 (震荡无法收敛)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print("审查结论：面对线性不可分数据，感知机无法找到完美超平面，损失函数持续震荡。")