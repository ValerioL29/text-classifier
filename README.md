# Text-Classifier

#### 介绍
🧐 - Textual Material Classifier

#### 软件架构
```
Text-Classifier
├─gbdt_classifier
├─naive_bayes
├─preprocesssvm_classifier
├─preprocess
├─resources
├─.gitignore
├─main.py
└─README.md
```

#### 安装教程

1.  进入项目目录文件夹，在终端执行 `pip install .` 创建venv
2.  或者创建 Conda 环境

#### 使用说明

1. 执行 `main.py` 文件中的程序入口
2. 等待推理，得到不同模型的对比分类结果

#### 实验说明

文本数据的分类与分析

##### 实验目的

* 掌握数据预处理的方法，对训练集数据进行预处理； 
* 掌握文本建模的方法，对语料库的文档进行建模； 
* 掌握分类算法的原理，基于有监督的机器学习方法，训练文本分类器； 
* 利用学习的文本分类器，对未知文本进行分类判别； 
* 掌握评价分类器性能的评估方法。 

##### 实验类型

数据挖掘算法的设计与编程实现。 

##### 实验要求

* 文本类别数：10类；
* 训练集文档数：>=50000篇；每类平均5000篇。 
* 测试集文档数：>=50000篇；每类平均5000篇。 
* 分组完成实验，组员数量<=3，个人实现可以获得实验加分。 

##### 实验内容

利用分类算法实现对文本的数据挖掘，主要包括：
1. 语料库的构建，主要包括利用爬虫收集Web文档等； 
2. 语料库的数据预处理，包括文档建模，如去噪，分词，建立数据字典，使用词袋模型或主题模型表达文档等；（注：使用主题模型，如LDA可以获得实验加分） 
3. 选择分类算法（朴素贝叶斯/SVM/其他等），训练文本分类器，理解所选的分类算法的建模原理、实现过程和相关参数的含义； 
4. 对测试集的文本进行分类 
5. 对测试集的分类结果利用正确率和召回率进行分析评价：计算每类正确率、召回率，计算总体正确率和召回率，以及F-score。 
 
##### 实验验收

1. 编写实验报告，实验报告内容必须包括对每个阶段的过程描述，以及实验结果的截图展示。 
2. 以线上方式验收实验。 
3. 实验完成时间6月20日. 
