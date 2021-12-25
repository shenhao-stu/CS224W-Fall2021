# Chapter4 GNN入门

- **斯坦福大学公开课CS224W的学习笔记**
- **Learner** : shenhao
- **Lecture 6，7**

## 4.0 导言

前面几节内容主要介绍了节点嵌入的概念，也就是说我们可以将一个图中的节点映射到一个的 $d$ 维向量上，而这种映射方式使得相似的节点对应的向量更接近，但主要的问题还是，我们如何学习出一个映射函数 $f$ ，而图嵌入的两个核心组件是**编码器**和**相似度**函数，之前也介绍了比较naive的shallow编码器，使用一个大矩阵直接储存每个节点的表示向量，通过矩阵与向量乘法来实现嵌入过程。
这种方法的**缺陷**在于：

- 需要 $O\left(|V|\right)$ 复杂度（参数量为嵌入向量的维度 $d$ × 节点数 $|V|$ ）
  - 节点间参数不共享
  - 每个节点都有它独自的嵌入向量

- 固有的“转换”
  - 无法获取在训练时没出现过的节点的嵌入向量
- 无法应用节点特征信息

而图神经网络GNN提供了一种基于深度学习的节点嵌入方法。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.0.png" style="zoom:33%;" />

**注意**：相比于图片和文本，网络结构要更复杂并且缺少规律性，而且往往是动态的。

## 4.1 图深度学习

### 4.1.1 定义

- 我们把要研究的图记为 $G$ ，图中所有节点构成集合 $V$ ，图的接邻矩阵记作 $A$ ，节点记作 $v$ ，节点 $v$ 的邻居节点集合记作 $N(v)$ 。
- 用 $\mathbf{X} \in \mathbb{R}^{m \times|V|}$ 表示图节点的特征矩阵，每个节点具有 $m$ 维的特征，节点的特征可以根据图的实际情况来选取。
  - 如果数据集中没有节点特征，可以用指示向量 indicator vectors （节点的独热编码），或者所有元素为常数1的向量，有时也会用节点度数来作为特征。

- 一种很naive的方法是**将图G的接邻矩阵和特征矩阵进行合并**，然后放到一个深度神经网络中进行学习，但是这样一来就会有$O(|V|)$数量级的参数，并且对于不同大小的图需要重新设计网络结构，这样的处理方式也使得结果对节点的顺序非常敏感。
- 我们需要的是一个即使改变了节点顺序，结果也不会变的模型。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.1.png" style="zoom:33%;" />

### 4.1.2 图神经网络与CNN

> 基本想法：将网格上的卷积神经网络泛化到图上，并应用到节点特征数据。

联想到CNN通过**卷积**的方式**来提取和融合邻近像素点的特征**，图结构中也可以聚合节点的邻近节点的特征，但是相比于图像，图结构往往不具有一个**固定的子结构**或者**滑动窗口**可以用来定义图中的卷积，并且是permutation invariant的，节点顺序是不固定的。

#### 4.1.2.1 设计思路	

因此，对于这些问题，我们可以**聚合局部的邻近节点的特征来生成图节点的嵌入向量**。

每个节点依据其周围邻居节点的信息生成一张**计算图**，通过前向传播和聚合函数进行信息的传递，最终计算出节点的嵌入。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.2.1_1.png" style="zoom: 50%;" />

这种神经网络可以是任意深度，每一层有节点的嵌入：

- 第0层**只用输入的特征表示节点的嵌入向量**
- 第k层是节点**通过聚合k hop邻居所形成的节点嵌入**。
- 每一层节点嵌入是聚合**上一层邻居节点的嵌入**再加上**它自己（相当于添加了自环）的嵌入**。 

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.2.1_2.png" style="zoom:33%;" />

#### 4.1.2.2 聚合函数

上述结构只是一种最基本的框架，最主要的区别在于图中的**box**对应的内容，也就是**信息聚合的方式**和之后的一系列处理。不同聚合方法的区别就在于**如何跨层聚合邻居节点信息**。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.2.2_1.png" style="zoom: 33%;" />

常见的方法：对得到的信息**求平均**并放到神经网络中

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.2.2_2.png" style="zoom:33%;" />
$$
\begin{equation}
\begin{aligned}
&\mathrm{h}_{v}^{0}=\mathrm{x}_{v} \\
&\mathrm{h}_v^{(l+1)}=\sigma (\mathrm{W}_l\sum_{u\in N(v)}\frac{\mathrm{h}_u^{(l)}}{|N(v)|}+\mathrm{B}_l\mathrm{h}_v^{(l)}) , \forall l \in\{0, \ldots, L-1\}\\
&\mathrm{z}_{v}=\mathrm{h}_{v}^{(L)} \\
\end{aligned}
\end{equation}
$$

- 第 0 层的嵌入向量 $\mathrm{h}_{v}^{0}$ 等于其节点特征向量 $\mathrm{x}_{v}$
- 这里的 $\sigma$ 是一个非线性的激活函数，eg：RELU
- $\sum_{u\in N(v)}\frac{\mathrm{h}_u^{(l)}}{|N(v)|}$ 是之前层的邻居节点的嵌入向量的均值
- $\mathrm{h}_u^{(l)}$ 是第 $l$ 层的节点 $v$ 的嵌入向量
-  $L$ 是层数
-  $\mathrm{z}_{v}$ 是经过 $L$ 层邻居聚合后的嵌入向量



**注意**：邻居信息聚合方法必须要**order invariant**或者说**permutation invariant**。
permutation invariant：如下图所示，计算结果应当不受到节点符号顺序的影响。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.2.2_3.png" style="zoom:33%;" />

在得到了嵌入向量之后可以**使用损失函数和随机梯度下降的方式来训练权重参数 $W$ 和 $B$ **

#### 4.1.2.3 向量化

我们可以将上面的表达式进行**向量化**来提高计算的效率。

- 令$H^{(l)}=[h_1^{(l)},\dots,h_{|V|}^{(l)}]^T$
  - 就有 $\sum_{u \in N_{v}} h_{u}^{(k)}=A_{v,:} \mathrm{H}^{(k)}$
- 对角矩阵 $D$ 来表示 $D_{v,v}=|N(v)|$
  - 就有 $D^{-1}_{v,v}=\frac 1 {N(v)}$ 

因此：
$$
\sum_{u \in N(v)} \frac{h_{u}^{(k-1)}}{|N(v)|} \Rightarrow H^{(k+1)}=D^{-1} A H^{(k)}
$$
这样一来原式就可以表示为：

$$
H^{(l+1)}=\sigma(D^{-1}A H^{(l)}W_l^T+H^{(l)}B_l^T)
$$

  -  $D^{-1}A H^{(l)}W_l^T$ ：邻居聚合的过程
  -  $H^{(l)}B_l^T$ ：自信息传递的过程



### 4.1.3 如何训练GNN

#### 4.1.3.1 无监督学习

无监督学习的训练方式：没有节点的标签，可以使用图的结构作为监督，定义一个loss函数并进行优化，**相似的节点会有相近的嵌入向量**，因此我们可以定义：
$$
\mathcal L=\sum_{z_u,z_v}\mathrm{CE}(y_{u,v},\mathrm {DEC}(z_u,z_v))
$$

-  $y$ 是表示两个节点相似的标签， $y_{u,v}=1$ 表示节点 $u$ 和 $v$ 相似
-  $\mathrm{CE}$ 表示交叉熵
-  $\mathrm {DEC}$ 是解码器，比如可以使用内积来定义
- 相似度的定义可以用前面提到的任何方法，包括随机游走，矩阵分解，图中节点的接近度等等

#### 4.1.3.2 监督学习

监督学习的训练方式：定义一个loss函数并进行优化。

对于一个节点分类的问题，可以用监督学习的方式直接训练模型，使用交叉熵loss表示如下：
$$
\mathcal L=\sum_{v\in V}y_v\log (\sigma(z_v^T\theta))+(1-y_v)\log (1-\sigma(z_v^T\theta))
$$

- 这里的 $y$ 表示节点的标签，而 $z$ 表示嵌入向量， $\theta$ 表示分类的权重

但是最终，我们学习到的一系列参数和模型是需要在新的图和新的节点上测试效果的，因此需要使模型**拥有更好的泛化能力** 

### 4.1.4 图神经网络模型的Pipeline

1. 定义一个邻居节点聚合的方式

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.4_1.png" style="zoom:25%;" />

2. 定义一个基于嵌入向量的损失函数

3. 使用一系列节点进行训练

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.4_2.png" style="zoom:25%;" />

4. 生成节点的嵌入，用于实际的任务中

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.4_3.png" style="zoom:25%;" />

### 4.1.5 泛化能力

因为聚合邻居的参数在所有节点之间共享，所以训练好的模型可以应用在没见过的节点/图上。比如动态图就有新增节点的情况。

- 模型参数数量是亚线性sublinear于 $|V|$ 的（仅取决于嵌入维度和特征维度）
- 矩阵尺寸就是下一层嵌入维度 × 上一层嵌入维度，第 0 层嵌入维度就是特征维度。

![](https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.5.png)

#### 4.1.5.1 图中的泛化能力

通过训练一张图的节点嵌入向量，将权重参数应用到一张新的图中，生成节点嵌入向量。

例子：在模型生物体A的蛋白质相互作用图上进行训练，并在新收集的生物体B的数据上生成嵌入。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.5.1.png" style="zoom:33%;" />

#### 4.1.5.2 新节点的泛化能力

同时，图神经网络的泛化能力还可以用来生成新节点的节点嵌入向量

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.1.5.2.png" style="zoom: 33%;" />

## 4.2 GNN概览

GNN由一系列GNN层线性组合构成，而GNN层包含了message和aggregation等多层次的信息，当我们训练一个GNN的时候可以从监督学习的视角出发进行训练，也可以从非监督学习的视角出发进行训练

在设计模型的过程中，可以选择的各种实现方式所组成的空间。比如说可以选择怎么卷积，怎么聚合，消息传递的机制，怎么将每一层网络叠起来，用什么激活函数、用什么损失函数，如何设计计算图，是否需要进行数据增强，图结构的其余操作等等。用这些选项组合出模型实例，构成的空间就是design space。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.png" style="zoom:33%;" />

### 4.2.1 GNN的单层结构

GNN的单层结构将多个向量组合成一个单独的向量，需要经过两个步骤的处理，分别是message和aggregation，将输入的节点嵌入向量转化成输出的节点嵌入向量

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.1_1.png" style="zoom:33%;" />

**message步骤**

每个节点都会创建一个消息发送给附近的一系列节点，这样一来就有
$$
m_u^{(l)}=\mathrm{MSG}^{(l)}(h_u^{(l-1)})
$$
eg：$\mathrm{m}_{u}^{(l)}=\mathrm{W}^{(l)} \mathrm{h}_{u}^{(l-1)}$

**aggregation步骤**

每个节点会收集从其他节点发来的消息并进行一定的处理，比如求和，求均值或者最大值等等
$$
h_v^{(l)}=\mathrm{ACG}^{(l)}(m_u^{(l)},u\in N(v))
$$
但是只有aggregation的话每一层的节点输出的嵌入向量都不**包含其自身从上一层携带回来的特征**，因此可以将message步骤和aggregation步骤得到的结果进行合并作为最终的输出结果，并且可以在两个过程中使用一些**非线性的激活函数**：

$$
\mathrm{m}_{u}^{(l)}=\mathrm{W}^{(l)} \mathrm{h}_{u}^{(l-1)} \\
\mathrm{m}_{v}^{(l)}=\mathrm{B}^{(l)} \mathrm{h}_{v}^{(l-1)} \\
h_v^{(l)}=\mathrm{CONCAT}(\mathrm {ACG}^{(l)}(m_u^{(l)},u\in N(v)), m_v^{(l)}) \\
$$

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.1_2.png" style="zoom:33%;" />

#### 4.2.1.1 GCN

对于**图卷积网络**GCN的层结构，上面已经介绍过了它每一层的嵌入向量更新方式
$$
h_v^{(l+1)}=\sigma (\sum_{u\in N(v)}W_l\frac{h_u^{(l)}}{|N(v)|})
$$

- 信息转换：
  - $\mathbf{m}_{u}^{(l)}=\frac{1}{|N(v)|} \mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}$
  - 对上一层的节点嵌入用本层的权重矩阵进行转换，用节点度数进行归一化（在不同GCN论文中会应用不同的归一化方式）
- 信息聚合：
  - $\mathbf{h}_{v}^{(l)}=\sigma\left(\operatorname{Sum}\left(\left\{\mathbf{m}_{u}^{(l)}, u \in N(v)\right\}\right)\right)$
  - 加总邻居信息，应用激活函数

#### 4.2.1.2 GraphSAGE

GCN是一种在图中结合拓扑结构和顶点属性信息学习顶点的embedding表示的方法。然而GCN要求在一个确定的图中去学习顶点的embedding，无法直接泛化到在训练过程没有出现过的顶点，即属于一种直推式(transductive)的学习。

而GraphSAGE是另一种架构的图神经网络，其核心思想是通过学习一个对邻居顶点进行聚合表示的函数来产生目标顶点的embedding向量。
$$
h_v^{(l)}=\sigma (W_l\times\mathrm{CONCAT}(B_lh_V^{(l-1)}, \mathrm{ACG}(h_u^{(l-1)},u \in N(v))))
$$

##### 运行流程

这种架构的Message是每个邻居节点发过来的嵌入向量，通过ACG函数进行aggregation，之后再和当前节点本身的。

信息转换在 AGG 过程中顺带着实现，而信息聚合分为两步：

- 第一步：聚合邻居节点信息
  - $h_{N(v)}^{(l)} \leftarrow \operatorname{AGG}\left(\left\{\mathbf{h}_{u}^{(l-1)}, \forall u \in N(v)\right\}\right)$ 
- 第二步：将上一步信息与节点本身信息进行聚合
  - $h_{v}^{(l)}=\sigma\left(\mathbf{W}^{(l)} \cdot \operatorname{CONCAT}\left(h_{v}^{(l-1)}, h_{N(v)}^{(l)}\right)\right.$

##### 聚合函数的设计

其中聚合函数ACG有多种不同的选择方式，比如使用均值函数，或者进行池化(最大池化，最小池化等等)，也可以使用LSTM来进行邻近节点的reshuffled。

1. Mean：邻居的加权平均值 

   $\mathrm{AGG}=\sum_{u \in N(v)} \frac{h_{u}^{(l-1)}}{|N(v)|}$

2. Pool：对邻居向量做转换，再应用对称向量函数，如求和 $\operatorname{Mean}(\cdot)$ 或求最大值 $\operatorname{Max}(\cdot)$ 

   $\mathrm{AGG}=\operatorname{Mean}\left(\left\{\operatorname{MLP}\left(h_{u}^{(l-1)}\right), \forall u \in N(v)\right\}\right)$

3. LSTM：在reshuffle的邻居上应用LSTM 

   $\mathrm{AGG}=\operatorname{LSTM}\left(\left[h_{u}^{(l-1)}, \forall u \in \pi(N(v))\right]\right)$

##### 归一化

在GraphSAGE每一层上都可以做 $\ell_{2}$ 归一化，即$h_v^{(l)}=\frac {h_v^{(l)}}{||h_v^{(l)}||_2}$ 。

使用了标准化操作之后就会使得所有嵌入向量的 $\ell_{2}$ 范数统一，可以给性能带来比较大的提升。

#### 4.2.1.3 GAT与注意力机制

GAT是Graph Attention Network，即在图神经网络中加入了**注意力机制**，我们可以使用 $\alpha_{vu}$ 来表示节点 $v$ 的邻居节点 $u$ 的重要程度，这样的网络可以使网络在计算时将注意力集中到一些比较重要的节点上面去。
$$
h_v^{(l)}=\sigma(\sum_{u\in N(v)}\alpha_{vu}W^{(l)}h_u^{(l-1)})
$$

在 GCN 和 GraphSAGE 中， $\alpha_{v u}=\frac{1}{|N(v)|}$ ，直接基于图结构信息 (节点度数) 显式定义注意力权 重，相当于认为节点的所有邻居都同样重要（注意力权重一样大）。

##### 注意力机制的计算方法

在GAT中注意力权重的计算方法是：通过attention mechanism $a$ (一个可训练的模型) 用两个节点上一层的节点嵌入计算两个**相邻节点的重要程度 $e_{v u}$ **，用 $e_{v u}$ 计算 $\alpha_{v u}$ 。

1. 首先定义一个计算重要性系数的函数 $a$ 并计算两个**相邻节点的重要程度 $e_{v u}$ **

$$
e_{vu}=a(W^{(l)}h_u^{(l-1)},W^{(l)}h_v^{(l-1)})
$$
<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/4.2.1.3_1.png" style="zoom:33%;" />

这个 $a$ 随便选 (可以是不对称的)，比如用单层神经网络，则 $a$ 有可训练参数 (线性层中的权重)：

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/4.2.1.3_2.png" style="zoom:33%;" />
$$
\begin{aligned}
&e_{vu}=a\left(\mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}, \mathbf{W}^{(l)} \mathbf{h}_{v}^{(l-1)}\right) \\
&=\text { Linear }\left(\text { Concat }\left(\mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}, \mathbf{W}^{(l)} \mathbf{h}_{v}^{(l-1)}\right)\right)
\end{aligned}
$$

2. 然后使用 softmax 来计算出权重注意力权重 $\alpha_{vu}$ ：

$$
\alpha_{v u}=\frac{\exp \left(e_{v u}\right)}{\sum_{k \in N(v)} \exp \left(e_{v k}\right)}
$$

3. 最后基于注意力权重 $\alpha_{vu}$ 进行邻居节点信息的加权求和：

$$
\mathbf{h}_{v}^{(l)}=\sigma\left(\sum_{u \in N(v)} \alpha_{v u} \mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}\right)
$$

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/4.2.1.3_3.png" style="zoom:33%;" />

##### 多头注意力机制

多头注意力机制：创建多个注意力分数并进行连接，增加模型鲁棒性，使模型不卡死在奇怪的优化空间，在实践上平均表现更好。
$$
h_v^{(l)}[i]=\sigma(\sum_{u\in N(v)}\alpha_{vu}^iW^{(l)}h_u^{(l-1)})
$$

举例：head=3

用不同参数建立多个 attention 模型:
$$
\begin{aligned}
&\mathbf{h}_{v}^{(l)}[1]=\sigma\left(\sum_{u \in N(v)} \alpha_{v u}^{1} \mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}\right) \\
&\mathbf{h}_{v}^{(l)}[2]=\sigma\left(\sum_{u \in N(v)} \alpha_{v u}^{2} \mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}\right) \\
&\mathbf{h}_{v}^{(l)}[3]=\sigma\left(\sum_{u \in N(v)} \alpha_{v u}^{3} \mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}\right)
\end{aligned}
$$
将输出进行聚合 (通过concatenation或加总) ：$\mathbf{h}_{v}^{(l)}=\mathrm{AGG}\left(\mathbf{h}_{v}^{(l)}[1], \mathbf{h}_{v}^{(l)}[2], \mathbf{h}_{v}^{(l)}[3]\right)$

##### 注意力机制的优点

最主要的优点是允许对不同的邻居节点采取不同的权重来突出一些重要的邻居

1. 计算高效：对attentional coefficients的计算可以在图中所有边上**并行**运算，聚合过程可以在所有节点上**并行**运算
2. 存储高效：稀疏矩阵运算需要存储的元素数不超过 $O(V+E)$ ，参数数目固定 $(a$ 的可训练参数尺寸与图尺寸无关)
3. 本地化：仅对本地网络邻居赋予权重
4. 具有推理能力：边间共享机制，与全局图结构无关，可以对新节点直接计算嵌入向量。

### 4.2.2 通用的GNN层设计

实践应用中的GNN网络层往往会应用传统神经网络模块，如在信息转换阶段应用Batch Normalization（使神经网络训练稳定）、Dropout（预防过拟合）、Attention / Gating（控制信息重要性）等。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.2_1.png" style="zoom: 33%;" />

- 批标准化：可以使得神经网络的训练更加稳定，做法就是求出每一层输出结果的平均值和方差，然后进行标准化处理，可以防止过大或者过小的数据出现而导致梯度消失或者梯度爆炸等问题

  <img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.2_2.png" style="zoom:33%;" />

- Dropout：用来避免神经网络过拟合，在训练的时候，以一定概率将一些神经元设置成0(即中途退出了计算)，在测试的时候使用所有的神经元参与到计算中。
  - 在GNN中一般在**线性层**使用dropout

- 非线性函数 / 激活函数：使用一些非线性的激活函数，将特征非线性化，常见的非线性激活函数有sigmoid，ReLU等等

  <img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.2_3.png" style="zoom:33%;" />

### 4.2.3 GNN的层级架构

上面主要介绍的是GNN单层的设计方式和基本的idea以及一些常用的神经网络tricks，但是在设计好了若干层神经网络之后，还需要用一定的规则将其组合起来，在GNN中往往就是将多个层级进行线性的组合，有的时候也会用的skip connection机制。

#### 4.2.3.1 过平滑问题

**过平滑问题**：是指当GNN层数太多的时候，所有节点的嵌入向量都将收敛到同一个值。这是一个非常糟糕的现象，因为我们希望不同的节点的嵌入向量是不同的，否则无法区别出不同的节点。

请注意：GNN的层跟别的神经网络的层不一样，**GNN的层数说明的是它聚集多少跳邻居的信息**。

为了解释过平滑问题的原因，在这里我们先引入**“感受野”**的概念

#### 4.2.3.2 感受野

**感受野(Receptive Field)** ：是决定一个感兴趣的节点的嵌入向量的节点集合

- 在一个 K 层的GNN中，每个节点的感受野包含了相距 K-hop 的邻居

- 可以用**感受野来解释过平滑问题**出现的原因

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.2_1.png" style="zoom:33%;" />

- 随着GNN的层数增加，每个节点的感受野也在不断扩大，这就会导致两个节点的感受野的重合度越来越高，而感受野可以决定一个节点的嵌入向量，因此随着层数增加嵌入向量也会越来越相似，最终就会导致过平滑问题的出现。
  - 堆叠很多GNN网络层→节点具有高度重合的感受野→节点嵌入高度相似→过平滑问题


<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.2_2.png" style="zoom:33%;" />

#### 4.2.3.3 解决过平滑问题

如何解决过平滑的问题？

在添加GNN层的时候慎重考虑，不像CNN之类的网络结构，GNN有的时候添加太多层可能不会起作用

- 因此需要先分析解决问题必要的感受野的大小
- 然后再将GNN层数设定为一个稍大于所需感受野的值，不能设置的太大，否则就会出现上述过平滑的问题

既然GNN层数不能太多，那么我们如何使一个浅的GNN网络更具有表现力呢？

- 增加单层GNN的表现力，如将信息转换/信息聚合过程从一个简单的线性网络变成深度神经网络（如3层MLP）

- 添加不是用来传递信息的网络层，也就是非GNN层，如对每个节点应用MLP（在GNN层之前或之后均可，分别叫 pre-process layers 和 post-process layers）

  - 前处理层 pre-processing layers：将节点特征进行编码转换为可以输入到GNN网络中的变量类型。（如节点表示为图像/文字时）

  - 后处理层 post-processing layers：在节点嵌入的基础上进行推理和转换（如图分类、知识图谱等任务中）

  <img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.3_1.png" style="zoom:25%;" />
  
    > colab2中的图分类任务就都有，用AtomEncoder作为pre-processing layer，池化层作为post-processing layer

- 增加skip connection，改变GNN层的连接方式

  - 有的时候位于更底层的GNN层产生的嵌入向量更能区分不同的节点，因此我们可以在最终节点嵌入中增加底层GNN层的影响力
  - 方法就是在GNN中添加一些捷径shortcuts，也叫做skip connection，来保存上一层节点的嵌入向量

  <img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.3_2.png" style="zoom:33%;" />


#### 4.2.3.4 skip connections原理

**为什么skip connection 可以work？**

因为skip机制创建了一个混合模型，相当于制造了多个模型（如图所示）， $N$ 个skip connections就相当于创造了 $2^N$ 条路径，每一条路径最多有 $N$ 个模块。

这些路径都会对最终的节点嵌入产生影响，使得比较底层的一些计算结果直接跳过了中间一些层的计算直接作用到了更高的层次中，这样一来就相当于自动获得了一个浅GNN和深GNN的融合模型。

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.4_1.png" style="zoom:33%;" />

因此，我们对GCN进行改进，引入有skip connection机制的模型，更新方式如下进行修改：

$$
h_v^{(l+1)}=\sigma (W_l\sum_{u\in N(v)}\frac{h_u^{(l)}}{|N(v)|})\rightarrow h_v^{(l+1)}=\sigma (W_l\sum_{u\in N(v)}\frac{h_u^{(l)}}{|N(v)|}+h_v^{(l)})
$$
<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.4_2.png" style="zoom:33%;" />

**补充**：skip connections也可以跨多层，直接跨到最后一层，在最后一层聚合之前各层的嵌入（通过 concat / pooling / LSTM ）

<img src="https://gitee.com/shenhao-stu/CS224W-Fall2021/raw/master/docs/doc_imgs/ch4/ch4.2.3.4_3.png" style="zoom:33%;" />

## 4.3 本章小结

通过本章的学习，我们了解了GNN网络的基本设计思路，以及介绍了三个传统的GNN网络，GCA，GraphSAGE和GAT。在下一章节，我们将深入探究图中的数据增强以及各类图任务的损失函数的设计和优化目标。
