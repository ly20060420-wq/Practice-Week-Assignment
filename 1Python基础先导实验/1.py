import random
import math
import matplotlib.pyplot as plt

# 示例 1：构建分类任务原生数据集（2 特征、2 分类）
# 单条样本格式：[特征列表, 标签]
dataset = [
 [[5.1, 3.5, 2.4], 0],
 [[4.9, 3.0, 3.3], 0],
 [[4.7, 3.2, 1.9], 0],
 [[4.6, 3.1, 2.2], 0],
 [[3.1, 5.5, 2.8], 0],
 [[6.4, 7.2, 4.1], 1],
 [[6.9, 4.2, 5.0], 1],
 [[5.5, 7.3, 4.6], 1],
 [[4.5, 6.6, 5.3], 1],
 [[7.0, 3.2, 4.8], 1]
]
# 示例 2：提取特征矩阵与标签向量
features = [sample[0] for sample in dataset]
labels = [sample[1] for sample in dataset]
print("特征矩阵：", features)
print("标签向量：", labels)

# 示例 3：数据集基础统计
n_samples = len(dataset)
n_features = len(features[0])
n_pos = sum(1 for label in labels if label == 1)
n_neg = n_samples - n_pos
print(f"样本总数：{n_samples}，特征维度：{n_features}")
print(f"正例数量：{n_pos},正例占比：{n_pos/(n_pos+n_neg)},\n负例数量：{n_neg},负例占比：{n_neg/(n_neg+n_pos)}")

# 示例 4：特征统计函数封装
def calc_feature_stats(features):
 n_features = len(features[0])
 stats = []
 for i in range(n_features):
    feature_values = [sample[i] for sample in features]
    mean_val = sum(feature_values) / len(feature_values)
    max_val = max(feature_values)
    min_val = min(feature_values)
    #计算每个特征的方差
    #平方和
    sum_=0.0
    for j in feature_values:
        sum_+=(j-mean_val)**2
    #样本/n-1 ，使用局部变量长度而不是全局 n_samples
    curr_n = len(feature_values)
    ans=sum_/(curr_n-1) if curr_n > 1 else 0.0
    
    stats.append({
        "特征索引": i,
        "均值": mean_val,
        "最大值": max_val,
        "最小值": min_val,
        "特征方差":ans
    })
 return stats

feature_stats = calc_feature_stats(features)
for stat in feature_stats:
    print(stat)

#筛选出第 0 个特征值大于 5 的所有样本
a=[]
for sample in dataset:
  if(sample[0][0]>5.0):
    a.append({
      "特征":sample[0],
      "标签":sample[1]
    })
print("第 0 个特征值大于 5 的所有样本:")
print(a)

#样本打乱,7:3划分
random.shuffle(dataset)#无返回值，直接修改原列表
train=int(n_samples*0.7)
train_dataset=dataset[:train]#取列表里面前train个元素
test_dataset=dataset[train:]#从列表里面下标为train的元素开始，取到最后
print(f"训练集的样本数量:{train},训练集样本：{train_dataset}\n测试集的样本数量:{n_samples-train},测试集样本：{test_dataset}")


# 实现按指定特征值对数据集进行升序/降序排序
def sort_dataset_by_feature(data, feature_idx, reverse=False):
    return sorted(data, key=lambda x: x[0][feature_idx], reverse=reverse)

print("\n按第 0 个特征降序排序后的数据集:")
sorted_dataset = sort_dataset_by_feature(dataset, 0, reverse=True)
for sample in sorted_dataset:
    print(sample)


#用字典结构重构数据集，每条样本包含样本 ID「特征列表」「标签」三个键值对
def rebuild_dict(dataset_):
  dataset_dict=[]
  #enumerate解包返回(索引, 元素([特征列表, 标签]))
  for i,sample in enumerate(dataset_):
    features_data=sample[0]
    label=sample[1]
    sample_dict={
      "ID":i,
      "特征列表":features_data,
      "标签":label
    }
    dataset_dict.append(sample_dict)
  return dataset_dict
   
dataset_dict=rebuild_dict(dataset)
print(f"字典结构重构后的数据集为：\n{dataset_dict}")

#从字典数据集中提取特征矩阵与标签向量
features_dict=[sample["特征列表"] for sample in dataset_dict]
labels_dict=[sample["标签"] for sample in dataset_dict]
print(f"字典数据集的特征矩阵：{features_dict}")
print(f"字典数据集的标签：{labels_dict}")

# 数据集基础统计
n_samples_dict=len(features_dict)
n_features_dict=len(features_dict[0])
n_pos_dict=sum(1 for label in labels_dict if label==1)
n_neg_dict=n_samples_dict-n_pos_dict
print(f"样本总数：{n_samples_dict}，特征维度：{n_features_dict}")
print(f"正例数量：{n_pos_dict},正例占比：{n_pos_dict/(n_pos_dict+n_neg_dict)},\n负例数量：{n_neg_dict},负例占比：{n_neg_dict/(n_neg_dict+n_pos_dict)}")

#特征方差计算
feature_stats_dict=calc_feature_stats(features_dict)
for stat in feature_stats_dict:
   print(stat)

#筛选出第 0 个特征值大于 5 的所有样本
b=[]
for sample in dataset_dict:
   if(sample["特征列表"][0]>5.0):
      b.append({
         "ID":sample["ID"],
         "特征列表":sample["特征列表"],
         "标签":sample["标签"]
      })
print("第 0 个特征值大于 5 的所有样本:")
print(b)

#样本打乱,7:3划分
random.shuffle(dataset_dict)
train_dict=int(n_samples_dict*0.7)
train_dataset_dict=dataset_dict[:train_dict]
test_dataset_dict=dataset_dict[train_dict:]
print(f"训练集的样本数量:{train_dict},训练集样本：{train_dataset_dict}\n测试集的样本数量:{n_samples_dict-train_dict},测试集样本：{test_dataset_dict}")


#模块二
# 示例 1：最小-最大归一化原生实现
def min_max_scaler(features):
    n_features = len(features[0])#在特征矩阵中提取特征数量
    scaled_features = []
    # 计算每个特征的极值
    feature_minmax = []
    for i in range(n_features):
        values = [sample[i] for sample in features]#提取所有样本的第i个特征值
        min_val = min(values)
        max_val = max(values)
        feature_minmax.append((min_val, max_val))
    # 逐样本缩放
    for sample in features:
        scaled_sample = []
        #逐特征
        for i in range(n_features):
            min_val, max_val = feature_minmax[i]
            scaled_val = 0.0 if max_val == min_val else (sample[i] - min_val) / (max_val - min_val)
            scaled_sample.append(scaled_val)#把归一化后的特征值加入当前样本
        scaled_features.append(scaled_sample)#把当前样本加入最终矩阵
    return scaled_features, feature_minmax

# 测试
scaled_features, minmax_params = min_max_scaler(features)
print("归一化后特征：", scaled_features)


#实现 Z-Score 标准化函数：
def zscore_standardize(features):
   n_feat_samples=len(features)
   n_features=len(features[0])
   scaled_features=[]
   feature_stats=[]#每个特征的均值和标准差
   for i in range(n_features):
      feature_values=[sample[i] for sample in features]
      mean_val=sum(feature_values)/n_feat_samples
      sum_val=sum((j-mean_val)**2 for j in feature_values)
      ans=sum_val/(n_feat_samples-1) if n_feat_samples > 1 else 0.0 
      std_val=ans**0.5
      feature_stats.append((mean_val,std_val))
   for sample in features:
      scaled_sample=[]
      for i in range(n_features):
         mean_val,std_val=feature_stats[i]
         scaled_val=0.0 if std_val==0 else (sample[i]-mean_val)/std_val
         scaled_sample.append(scaled_val)
      scaled_features.append(scaled_sample)
   return scaled_features,feature_stats

def calc_feature_stats_v2(features):
 n_features = len(features[0])
 stats = []
 for i in range(n_features):
    feature_values = [sample[i] for sample in features]
    mean_val = sum(feature_values) / len(feature_values)
    #计算每个特征的方差
    #平方和
    sum_=0.0
    for j in feature_values:
        sum_+=(j-mean_val)**2
    #样本/n-1 
    curr_n = len(feature_values)
    ans=sum_/(curr_n-1) if curr_n > 1 else 0.0
    stats.append({
        "均值": mean_val,
        "特征方差":ans
    })
 return stats

#测试
scaled_features,feature_stats=zscore_standardize(features)
print(f"(1)标准化后的特征矩阵:{scaled_features}\n(1)每个特征的均值和标准差:{feature_stats}")

feature_stats_pre = calc_feature_stats_v2(features)
print(f"(3)标准化前的特征均值与方差:")
for stat in feature_stats_pre:
    print(stat)         


# 示例 2：缺失值填充原生实现
def fill_missing_value(features, fill_strategy="median"):
    n_features = len(features[0])
    filled_features = [sample.copy() for sample in features]#sample.copy()先填充原特征矩阵
    # 计算每个特征维度的填充值
    fill_values = []
    for i in range(n_features):
        valid_values = [sample[i] for sample in features if sample[i] is not None]
        if fill_strategy == "mean":
            fill_val = sum(valid_values) / len(valid_values)
        elif fill_strategy == "median":
            sorted_vals = sorted(valid_values)
            n = len(sorted_vals)
            fill_val = sorted_vals[n//2] if n%2 ==1 else (sorted_vals[n//2-1]+sorted_vals[n//2])/2
            #"//"向下取整
        else:
            raise ValueError("可选填充策略：mean/median")
        fill_values.append(fill_val)
    # 执行填充
    for sample in filled_features:
        for i in range(n_features):
            if sample[i] is None:
                sample[i] = fill_values[i]
    return filled_features, fill_values

# 测试：实现中位数填充函数，完成缺失值处理，打印填充前后的特征矩阵。
features_with_nan = [[5.1, None], [4.9, 3.0], [None, 3.2], [6.4, 3.2]]
print("填充前特征矩阵：", features_with_nan)
filled_features, fill_vals = fill_missing_value(features_with_nan, fill_strategy="median")
print("中位数填充后特征矩阵：", filled_features)



#实现离散特征的标签编码函数
def labels_code(features):
   fliter_=[]#对原始列表中的元素去重
   for sample in features:
      if sample not in fliter_:
         fliter_.append(sample)
   fliter_dict={}#构建映射字典
   for i,sample in enumerate(fliter_):
      fliter_dict[sample]=i
   encoded_list=[]
   for sample in features:
      encoded_list.append(fliter_dict[sample])#用字典的键把文字转化成数字
   return encoded_list,fliter_dict

#测试
weather=["Sunny", "Rainy", "Sunny", "Cloudy"]
encoded_list,map_dict=labels_code(weather)
print(f"原始列表：{weather}\n'标签'编码之后的数值列表：{encoded_list}\n编码映射字典：{map_dict}")


#实现离散特征的独热编码函数
def one_hot_encoding(features):
   fliter_=[]
   for sample in features:
      if sample not in fliter_:
         fliter_.append(sample)
   map_dict={}
   for i,sample in enumerate(fliter_):
      map_dict[sample]=i
   encoded_list=[]
   n_code=len(fliter_)
   for sample in features:
      temp_list=[0]*n_code
      temp_list[map_dict[sample]]=1
      encoded_list.append(temp_list)
   return encoded_list,map_dict

#测试
weather_hot_encode=["Sunny", "Rainy", "Sunny", "Cloudy"]
encoded_list_hot,map_dict_hot=one_hot_encoding(weather_hot_encode)
print(f"\n原始列表：{weather_hot_encode}\n'独热'编码之后的数值列表：{encoded_list_hot}\n编码映射字典：{map_dict_hot}")


#实现预处理流程链式调用
def preprocess(features,labels,fill_strategy="mean",standardize_method="zscore",train_propotion=0.7):
   #缺失值填充
   filled_features,fill_values=fill_missing_value(features,fill_strategy)
   #标准化
   if standardize_method=="minmax":
      scaled_features,feature_minmax=min_max_scaler(filled_features)#这里传上一步的filled_features，不是原始features
   elif standardize_method=="zscore":
      scaled_features,feature_stats=zscore_standardize(filled_features)
   else:
      raise ValueError("仅可选择minmax/zscore进行标准化")
   #数据集划分
   #经过上面处理，features变成了新的scaled_features,不能直接使用原来的dataset
   combined=list(zip(scaled_features,labels))
   random.shuffle(combined)
   train=int(len(combined)*train_propotion)
   train_dataset=combined[:train]
   test_dataset=combined[train:]

   return train_dataset,test_dataset

#测试
train_dataset,test_dataset=preprocess(features,labels,fill_strategy="mean",standardize_method="minmax")
print(f"预处理流程链式调用之后，训练集：{train_dataset}\n测试集：{test_dataset}")


#模块3
#机器学习核心逻辑的函数封装与模块化实现
# 模块 1：计算欧式距离
def calc_euclidean_distance(x1, x2):
    if len(x1) != len(x2):
        raise ValueError("两个样本的特征维度必须一致")
    squared_sum = 0.0
    for a, b in zip(x1, x2):#把2个样本的特征一一配对
        squared_sum += (a - b) ** 2
    return math.sqrt(squared_sum)

#计算曼哈顿距离
def calc_manhattan_distance(x1,x2):
   if len(x1)!=len(x2):
      raise ValueError("两个样本的特征维度必须一致")
   distance=0.0
   for a,b in zip(x1,x2):
      distance+=abs(a-b)
   return distance

# 模块 2：获取 K 个最近邻
def get_k_neighbors(train_set, test_sample, k,distance="euclidean"):
    distances = []
    for train_sample in train_set:
        train_features = train_sample[0]
        train_label = train_sample[1]
        if distance=="euclidean":
           dist=calc_euclidean_distance(train_features,test_sample)
        elif distance=="manhattan":
           dist=calc_manhattan_distance(train_features,test_sample)
        else:
           raise ValueError("距离仅能选择euclidean/manhattan")
        distances.append((train_sample, dist))
    # 按距离(x[1]是距离，x[0]是训练样本)升序排序
    distances.sort(key=lambda x: x[1])#lambda x:把列表里的每一个元素取名x,取x[1]位置的值
    # 选取前 K 个近邻
    neighbors = [(item[0],item[1]) for item in distances[:k]]#带权重的话得返回item[1]距离
    return neighbors

# 模块 3：投票预测标签
def predict(neighbors):
    label_votes = {}
    for neighbor in neighbors:
        label = neighbor[0][1]#不带权重时neighbors:[[特征]，标签]，带权重:([[特征]，标签]，距离)
        label_votes[label] = label_votes.get(label, 0) + 1
        #.get(键,默认值)，去字典里拿label对应的票数，如果没有这个标签，票数默认是0，把票数+1再存回字典
    # 返回票数最高的标签
    sorted_votes = sorted(label_votes.items(), key=lambda x: x[1], reverse=True)#字典无法指定按票数排序，.items()把字典变成(标签,票数)的列表
    return sorted_votes[0][0]

#带权重的投票预测(距离越近的样本投票权重越高（权重为距离的倒数）)
def predict_weight(neighbors):
   label_weights={}
   for neighbor,dist in neighbors:
      label=neighbor[1]
      weight=1.0/(dist + 1e-5) #加 1e-5 防止距离为 0 时出现除零报错
      label_weights[label]=label_weights.get(label,0)+weight
   sorted_votes=sorted(label_weights.items(),key=lambda x:x[1],reverse=True)
   return sorted_votes[0][0]

# 模块 4：KNN 完整流程封装
def knn_classifier(train_set, test_set, k=3,distance="euclidean",weight=False):
    predictions = []
    for test_sample in test_set:
        test_features = test_sample[0]
        neighbors = get_k_neighbors(train_set, test_features, k,distance)
        if weight:
            pred_label = predict_weight(neighbors)
        else:
            pred_label = predict(neighbors)
        predictions.append(pred_label)
    return predictions


# 测试，将数据集按 7:3 划分为训练集与测试集,对训练集和测试集进行标准化预处理
train_set,test_set=preprocess(features,labels,standardize_method="zscore",train_propotion=0.7)
print(f"数据集7:3划分,并进行z-score标准化预处理之后:\n训练集：{train_set}\n测试集：{test_set}")

#分别设置 k=1、k=3、k=5，调用 KNN 分类器完成预测，输出每组 k 值对应的预测标签与真实标签；
true_labels=[s[1] for s in test_set]
print(f"KNN分类器,欧式距离:")
for k in [1,3,5]:
   pred_labels=knn_classifier(train_set,test_set,k,distance="euclidean")
   print(f"k={k}时，真实标签：{true_labels}, 预测标签：{pred_labels}")

print(f"KNN分类器,曼哈顿距离:")
for k in [1,3,5]:
   pred_labels=knn_classifier(train_set,test_set,k,distance="manhattan")
   print(f"k={k}时，真实标签：{true_labels}, 预测标签：{pred_labels}")

print("带权重KNN,欧式距离,权重=距离倒数:")
for k in [1,3,5]:
   pred_weighted = knn_classifier(train_set, test_set, k=3, distance="euclidean", weight=True)
   print(f"k={k}时，真实标签：{true_labels}, 预测标签：{pred_weighted}")



#封装 KNN 算法为 Python 类，包含__init__（设置 k 值、距离类型）、fit（存储训练集）、predict（预测）三个核心方法，完成面向对象编程实现
class KNNClassifier:
   def __init__(self,k=3,distance="euclidean",weight=False):
      self.k=k
      self.distance=distance
      self.weight=weight
      self.train_set=None

   def fit(self,train_set):
      self.train_set=train_set

   def calc_euclidean_distance(self,x1,x2):
         if len(x1)!=len(x2):
            raise ValueError("两个样本的特征维度必须一致")
         squared_sum=0.0
         for a,b in zip(x1,x2):
            squared_sum+=(a-b)**2
         return math.sqrt(squared_sum)
      
   def calc_manhattan_distance(self,x1,x2):
         if len(x1)!=len(x2):
            raise ValueError("两个样本的特征维度必须一致")
         distance=0.0
         for a,b in zip(x1,x2):
            distance+=abs(a-b)
         return distance
      
   def get_k_neighbors(self,test_sample):
         distance=[]
         for train_sample in self.train_set:
            train_features=train_sample[0]
            train_labels=train_sample[1]
            if self.distance=="euclidean":
               dist=self.calc_euclidean_distance(train_features,test_sample)
            elif self.distance=="manhattan":
               dist=self.calc_manhattan_distance(train_features,test_sample)
            else:
               raise ValueError("距离仅能选择euclidean/manhattan")
            distance.append((train_sample,dist))#外层要再加一层括号，否则TypeError: list.append() takes exactly one argument (2 given)
         distance.sort(key=lambda x:x[1])
         neighbors=[(item[0],item[1]) for item in distance[:self.k]]
         return neighbors
      
   def predict_weight(self,neighbors):
         label_weight={}
         for neighbor,dist in neighbors:
            label=neighbor[1]
            weight=1.0/(dist + 1e-5)
            label_weight[label]=label_weight.get(label,0)+weight
         sorted_votes=sorted(label_weight.items(),key=lambda x:x[1],reverse=True)
         return sorted_votes[0][0]

   def predict_unweight(self,neighbors):
         label_votes={}
         for neighbor,dist in neighbors:
            label=neighbor[1]
            label_votes[label]=label_votes.get(label,0)+1
         sorted_votes=sorted(label_votes.items(),key=lambda x:x[1],reverse=True)
         return sorted_votes[0][0]
      
   def predict(self,test_set):
         if self.train_set is None:
            raise ValueError("请先调用fit方法存储训练集")
         predictions=[]
         for test_sample in test_set:
            test_features=test_sample[0]
            test_labels=test_sample[1]
            neighbors=self.get_k_neighbors(test_features)
            if self.weight:
               pre_label=self.predict_weight(neighbors)
            else:
               pre_label=self.predict_unweight(neighbors)
            predictions.append(pre_label)
         return predictions
      
#封装测试
train_set = [
    [[5.1, 3.5], 0], [[4.9, 3.0], 0], [[4.7, 3.2], 0],
    [[7.0, 3.2], 1], [[6.4, 3.2], 1], [[6.9, 3.1], 1]
]
test_set = [[[5.0, 3.4], 0], [[6.5, 3.0], 1]]
#实例化，设置超参数
knn_model=KNNClassifier(k=3,distance="euclidean",weight=True)
#训练
knn_model.fit(train_set)
#预测
preds=knn_model.predict(test_set)
print(f"封装后的KNN分类器,欧式距离,权重=距离倒数:")
print("预测标签：", preds)
print("真实标签：", [sample[1] for sample in test_set])



#模块四：模型评估指标与结果可视化实现
# 示例 1：分类评估指标原生实现
def calc_classification_metrics(y_true, y_pred, pos_label=1):
    # 计算混淆矩阵元素
    TP = TN = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == pos_label and pred == pos_label:
            TP += 1
        elif true != pos_label and pred != pos_label:
            TN += 1
        elif true != pos_label and pred == pos_label:
            FP += 1
        elif true == pos_label and pred != pos_label:
            FN += 1
    # 计算评估指标
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total !=0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) !=0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) !=0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) !=0 else 0.0
    # 返回结果
    return {
        "混淆矩阵": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "准确率": accuracy,
        "精确率": precision,
        "召回率": recall,
        "F1 值": f1
    }
# 测试
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0]
metrics = calc_classification_metrics(y_true, y_pred)

print(f"示例模型效果测试：\n真实标签：{y_true}\n预测标签：{y_pred}")
for key, value in metrics.items():
    print(f"{key}: {value}")

# 分析k值对模型效果的影响
dataset_expand = [
    [[2.7, 3.4, 2.8], 0], [[1.6, 4.2, 3.1], 0],
    [[4.2, 4.3, 1.4], 0], [[6.6, 6.5, 4.1], 1],
    [[6.8, 7.7, 4.7], 1], [[3.4, 3.9, 3.7], 0],
    [[7.5, 6.3, 4.3], 1], [[3.7, 1.7, 1.4], 0],
    [[5.1, 5.1, 4.9], 1], [[5.7, 2.4, 2.1], 0],
    [[6.5, 6.1, 4.8], 1], [[4.3, 4.1, 3.0], 0],
    [[6.1, 5.9, 4.0], 1], [[5.7, 1.8, 3.5], 0],
    [[3.9, 3.3, 2.4], 0], [[4.6, 2.9, 1.9], 0],
    [[3.5, 2.9, 2.5], 0], [[4.9, 3.3, 2.9], 0],
    [[6.4, 4.2, 4.4], 1], [[7.8, 5.4, 3.8], 1],
    [[3.6, 3.8, 3.3], 0], [[7.2, 6.8, 4.2], 1],
    [[3.6, 5.1, 2.6], 0], [[6.8, 4.3, 3.7], 1],
    [[7.1, 4.5, 5.0], 1], [[4.6, 2.0, 0.1], 0],
    [[4.1, 2.3, 2.7], 0], [[4.7, 4.1, 4.8], 0],
    [[4.6, 5.4, 4.4], 1], [[8.4, 5.5, 6.7], 1],
    [[6.6, 6.0, 5.7], 1], [[4.0, 3.4, 3.0], 0],
    [[4.3, 3.2, 2.3], 0], [[5.4, 4.3, 3.2], 0],
    [[3.4, 2.6, 3.4], 0], [[4.5, 5.8, 4.9], 1],
    [[2.5, 3.2, 3.0], 0], [[7.0, 6.0, 5.0], 1],
    [[5.8, 5.9, 6.2], 1], [[5.8, 5.6, 4.2], 1],
    [[7.2, 5.2, 4.4], 1], [[5.2, 3.5, 4.5], 0],
    [[2.8, 4.4, 2.5], 0], [[6.0, 3.8, 4.8], 1],
    [[7.3, 4.2, 2.2], 1], [[6.5, 7.8, 2.6], 1],
    [[4.5, 3.2, 4.8], 0], [[3.5, 4.5, 1.1], 0],
    [[3.2, 4.5, 1.9], 0], [[6.5, 4.8, 5.0], 1],
    [[6.2, 5.9, 6.0], 1], [[4.7, 3.4, 1.0], 0],
    [[7.2, 6.8, 5.1], 1], [[7.0, 4.9, 4.9], 1],
    [[7.9, 5.5, 3.4], 1], [[7.5, 4.8, 4.9], 1],
    [[6.5, 6.0, 5.1], 1], [[5.0, 5.0, 5.0], 1],
    [[4.0, 3.8, 2.7], 0], [[6.7, 4.9, 5.4], 1],
    [[6.0, 4.9, 2.3], 1], [[4.7, 2.3, 3.3], 0],
    [[4.1, 4.3, 3.1], 0], [[6.3, 4.0, 5.0], 1],
    [[6.5, 5.0, 3.9], 1], [[3.4, 3.8, 0.9], 0],
    [[5.8, 4.8, 6.0], 1], [[7.9, 7.6, 5.1], 1],
    [[3.5, 2.4, 2.3], 0], [[7.5, 6.0, 7.8], 1],
    [[6.7, 5.7, 4.1], 1], [[6.3, 3.8, 3.9], 0],
    [[7.0, 3.5, 4.5], 1], [[4.1, 3.7, 3.7], 0],
    [[5.3, 6.1, 2.2], 1], [[4.0, 3.8, 2.2], 0],
    [[5.4, 3.4, 1.1], 0], [[5.1, 4.4, 2.8], 0],
    [[6.6, 7.3, 5.4], 1], [[6.0, 4.3, 3.7], 1],
    [[5.8, 6.6, 4.6], 1], [[6.8, 8.7, 3.6], 1],
    [[6.3, 7.8, 6.0], 1], [[3.0, 3.4, 2.2], 0],
    [[3.0, 4.4, 2.9], 0], [[3.9, 4.2, 1.8], 0],
    [[4.6, 5.0, 3.2], 0], [[6.8, 4.7, 3.3], 1],
    [[4.1, 3.0, 1.3], 0], [[5.4, 3.1, 2.1], 0],
    [[3.0, 3.7, 3.8], 0], [[3.4, 2.8, 2.0], 0],
    [[6.2, 5.6, 6.0], 1], [[3.2, 2.4, 3.3], 0],
    [[6.3, 7.2, 3.5], 1], [[4.9, 5.1, 5.7], 1],
    [[4.2, 3.3, 4.5], 0], [[4.7, 3.6, 1.8], 0],
    [[6.7, 3.4, 6.5], 1], [[5.4, 4.0, 3.3], 0]
]
#固定全局随机种子，保证每次运行打乱的数据顺序完全一致，方便发现实验规律
random.seed(42)

print(f"基于KNN预测结果，分析k值对模型效果的影响")
features_raw=[sample[0] for sample in dataset_expand]
labels_raw=[sample[1] for sample in dataset_expand]

train_set,test_set=preprocess(features_raw,labels_raw,fill_strategy="mean",standardize_method="zscore",train_propotion=0.65)
y_true=[sample[1] for sample in test_set]
accuracies=[]
predictions=[]
k_values=[1,3,5]
k_values_expand=[1,3,5,7,9]
for k in k_values:
   knn=KNNClassifier(k=k,distance="euclidean",weight=False)
   knn.fit(train_set)
   y_pred=knn.predict(test_set)
   predictions.append(y_pred)
   metrics=calc_classification_metrics(y_true,y_pred)
   accuracies.append(metrics["准确率"])#取出metrics字典里"准确率"对应的值
   print(f"k={k}时")
   for key,value in metrics.items():
      print(f"{key}:{value}")

# 示例 2：预测结果可视化
#绘制不同 k 值对应的准确率变化折线图
plt.rcParams['font.sans-serif'] = ['SimHei']  #用黑体
plt.rcParams['axes.unicode_minus'] = False    #正常显示负号

def plot_accuracy(k_values,accuracies):
    plt.figure(figsize=(6, 4))#设置画布大小
    plt.plot(k_values,accuracies,marker="o",linestyle="-",color="blue",markersize=8)
    ## k_values: X轴数据，accuracies: Y轴数据，"o": 数据点用圆圈标记，"-": 线的样式为实线，markersize=8: 标记点的大小为8
    plt.xlabel("k 值")
    plt.ylabel("准确率 (Accuracy)")
    plt.title("KNN 不同 k 值下的准确率变化")
    plt.xticks(k_values)#设置刻度
    plt.yticks([0.0, 1.1])
    plt.grid(True, linestyle="--", alpha=0.7)#"--"：虚线，alpha：网格线透明度
    plt.show()
plot_accuracy(k_values,accuracies)

#绘制测试集样本的真实与预测标签对比散点图
def plot_scatter_comparision(y_true,y_pred,k_val):
    plt.figure(figsize=(6,4))
    x=range(len(y_true))
    plt.scatter(x,y_true,label="真实标签",color="blue",marker="o",s=80)
    plt.scatter(x,y_pred,label="预测标签",color="red",marker="x",s=80)
    plt.xlabel("测试集样本索引")
    plt.ylabel("分类标签")
    plt.title(f"真实标签与预测标签对比,k={k_val}")
    plt.xticks(x)
    plt.yticks([0,1])
    plt.legend()#显示图例
    plt.grid(True,linestyle="--",alpha=0.7)
    plt.show()
plot_scatter_comparision(y_true,predictions[1],k_val=3)

#绘制混淆矩阵热力图
def plot_confusion_matrix_heatmap(y_true,y_pred):
   cm=[[0,0],[0,0]]#[[TN,FP],[FN,TP]]
   for true_label,pred_label in zip(y_true,y_pred):
      cm[true_label][pred_label]+=1
   print(f"混淆矩阵：\n{cm}")

   fig,ax=plt.subplots(figsize=(6,4))#ax: 坐标轴对象
   cax=ax.imshow(cm,cmap="Blues")#cax接收图像对象，imshow把二维数组变成图像
   fig.colorbar(cax)#画布旁边添加颜色条
   ax.set_xticks([0,1])
   ax.set_yticks([0,1])
   ax.set_xticklabels(["预测0负例","预测1正例"])
   ax.set_yticklabels(["真实0负例","真实1正例"])
   plt.title("混淆矩阵热力图")
   ax.set_xlabel("预测标签")
   ax.set_ylabel("真实标签")

   for i in range(2):
      for j in range(2):#在坐标(j,i)处写入cm[i][j]，ha="center"水平居中，va垂直居中，fontsize字体大小
         ax.text(j,i,cm[i][j],ha="center",va="center",color="black",fontsize=12)
   plt.show()

#测试
test_y_true = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
test_y_pred = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
plot_confusion_matrix_heatmap(test_y_true, test_y_pred)


#回归任务核心评估指标：均方误差（MSE）、平均绝对误差（MAE）、决定系数 R²
def calc_regression_metrics(y_true, y_pred):
    n = len(y_true)
    mean_y = sum(y_true)/n
    mse_sum=0.0
    mae_sum=0.0
    all_sum=0.0  #总平方和，真实值和平均值的差异
    for true_val,pred_val in zip(y_true,y_pred):
        error=true_val-pred_val#算偏差
        mse_sum+=error ** 2
        mae_sum+=abs(error)
        all_sum+=(true_val - mean_y) ** 2
    mse=mse_sum/n
    mae=mae_sum/n
    if all_sum==0:
        r2=0.0 
    else:
        # R²=1-(模型误差/平均值的误差)
        r2=1.0-(mse_sum/all_sum)     
    return {"MSE":mse,"MAE":mae,"R2":r2}

#测试
y_true_1 = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred_1 = [1.3, 1.0, 3.0, 4.1, 5.0] 
metrics_1 = calc_regression_metrics(y_true_1, y_pred_1)
for k,v in metrics_1.items():
    print(f"{k}: {v}")