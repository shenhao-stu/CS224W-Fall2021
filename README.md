<div align=center>
<img src="https://raw.githubusercontent.com/shenhao-stu/CS224W-Fall2021/master/Assets/Stanford.png" width="250">
</div>
<p align="center">CS224W | shenhao0223@163.sufe.edu.cn | 上海财经大学 </p>

# DataWhale CS224W 学习路线

课号：[CS224W](http://web.stanford.edu/class/cs224w/)

教授：[Jure Leskovec](https://profiles.stanford.edu/jure-leskovec)

- [x] [bilibili](https://www.bilibili.com/video/BV1RZ4y1c7Co/?spm_id_from=333.788.recommend_more_video.0)，[精翻](https://www.bilibili.com/video/BV1Qq4y1f7tt?p=1)
- [x] [Lab x 5](http://web.stanford.edu/class/cs224w/projects.html)
- [x] [Complete Notes & Slides](http://web.stanford.edu/class/cs224w/index.html#schedule)

## 课程信息

复杂数据可以表示为对象之间的关系图。 而这类网络是建模社会，技术和生物系统的基本工具。这门课程主要聚焦分析大量图时所面对的计算、算法和建模挑战。通过研究底层图结构及其特征，学习者可以了解机器学习技术和数据挖掘工具，从而在多种网络中有所发现。

这门课程涉及的主题包括：表征学习和图神经网络；万维网算法；基于知识图谱的推理；影响力最大化；疾病爆发检测；社交网络分析。

## 适合人群

接触过深度学习，但想更进一步的入门图神经网络的同学。

## 先修条件

笔者认为只要或多或少接触过深度学习这门课程的同学都能完整的学完CS224W，并有自己的感悟。 

以下是官方的prerequisites.  

> Students are expected to have the following background:
>
> - Knowledge of basic computer science principles, sufficient to write a reasonably non-trivial computer program (e.g., CS107 or CS145 or equivalent are recommended)
> - Familiarity with the basic probability theory (CS109 or Stat116 are sufficient but not necessary)
> - Familiarity with the basic linear algebra (any one of Math 51, Math 103, Math 113, or CS 205 would be much more than necessary)
>
> The recitation sessions in the first weeks of the class will give an overview of the expected background.

## 食用方法

:whale: 推荐使用 [**CS224W Github在线阅读**](https://shenhao-stu.github.io/CS224W-Fall2021/) 进行学习。

CS224W Fall 2021的课程Slides，可以在项目`Slides`目录的下进行下载。

配套作业和答案，可以在项目`Assigments`和`Codes`目录的下进行学习。

## Syllabus

| 文件名                                                       | Lecture                                      |
| ------------------------------------------------------------ | -------------------------------------------- |
| [CS224W 图机器学习01](https://shenhao-stu.github.io/CS224W-Fall2021/#/ch1_图机器学习导论) | 导论，传统的图学习方法（Lecture1，Lecture2） |
| [CS224W 图机器学习02](https://shenhao-stu.github.io/CS224W-Fall2021/#/ch2_随机游走算法及PageRank) | 节点嵌入和PageRank（Lecture3，Leture4）      |
| [CS224W 图机器学习03](https://shenhao-stu.github.io/CS224W-Fall2021/#/ch3_消息传递和节点分类) | 消息传递机制                                 |
| [CS224W 图机器学习04](https://shenhao-stu.github.io/CS224W-Fall2021/#/ch4_GNN入门) | 图神经网络基础                               |
| CS224W 图机器学习05                                          | 图神经网络的训练和应用，表示能力分析         |
| CS224W 图机器学习06                                          | 知识图谱，知识图谱推理                       |
| CS224W 图机器学习07                                          | 频繁子图挖掘                                 |
| CS224W 图机器学习08                                          | 社区检测                                     |
| CS224W 图机器学习09                                          | 图生成模型                                   |
| CS224W 图机器学习10                                          | 图神经网络前沿话题                           |



## 课程评价

课程主讲人 Jure Leskovec 是斯坦福大学计算机科学副教授，也是图表示学习方法 node2vec 和 GraphSAGE 的作者之一。他主要的研究兴趣是社会信息网络的挖掘和建模等，特别是针对大规模数据、网络和媒体数据。据 Google Scholar 显示，Jure Leskovec 发表论文 400 余篇，被引用次数超过 82000 次，h 指数为 114。其论文多次发表在 Nature、NeurIPS、KDD、ICML 等期刊和学术会议上，并两次获得 KDD 时间检验奖。

作为学术大牛，Leskovec教授的授课质量也十分高，除了一点点“俄式”口音，但是配合中英字幕也可以无压力学习。

其次，课程作业也设计很好，贴合课程内容，能够加深对内容的学习，只需要一些深度学习pytorch编程的基础，就可以按照要求设计一个图神经网络。这门课的作业使用的是[PyTorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric)。  

> PS：这个工具的作者也是leskovec的学生~ 

## 非官方资料推荐

- [图神经网络（Graph Neural Networks，GNN）综述](https://zhuanlan.zhihu.com/p/75307407)
- [edx Mooc](https://www.edx.org/course/advanced-algorithmics-and-graph-theory-with-python)：Advanced Algorithmics and Graph Theory with Python
- 百度AI Studio 《[图神经网络7日打卡营](https://aistudio.baidu.com/aistudio/education/group/info/1956)》
- [Paddle Graph Learning (PGL) ](https://github.com/PaddlePaddle/PGL)：基于 PaddlePaddle 的高效易用的图学习框架
- maelfabien大佬的Github仓库：[Machine Learning Tutorials and Articles](https://github.com/maelfabien/Machine_Learning_Tutorials)
- [李沐 | 零基础多图详解图神经网络（GNN/GCN）【论文精读】](https://www.bilibili.com/video/BV1iT4y1d7zP)
- [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
