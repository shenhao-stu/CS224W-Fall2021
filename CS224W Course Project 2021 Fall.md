<center><h1>
CS 224W Project (2021 Fall)
</h1></center>
<center><h3>
Tutorials and Case Studies for Applying Graph ML to Real-World Problems
</h3></center>
[TOC]
# **Overview**

For this year’s CS224W course project, we want to develop a set of tutorials and case studies of applying Graph Machine Learning to a given problem domain. The goal is to create a long-lasting resource, which will help the community and further popularize the field of graph machine learning and PyG/GraphGym GNN library in particular.



We ask you to create a [Medium](https://www.google.com/url?q=https://medium.com/&sa=D&source=editors&ust=1637991494011000&usg=AOvVaw2Im5uBU_3piJVhXKHUDVOM)-based-blog tutorial/case study on applying state-of-the-art graph ML to a real-world application. Making a blog tutorial is a great exercise for you to better understand and implement graph ML algorithms to solve real problems. Moreover, your graph ML blog will become a great resource for the broader community to study the emergent field of graph ML. After our reviews, you can also share your blog posts on your own websites / social media accounts to impress your peers! Students can form a group of up to 3 people to work on the project. We are going to provide a place for you to publish a blog post.



For the implementation tool, we highly recommend using the [PyG](https://www.google.com/url?q=https://pytorch-geometric.readthedocs.io/en/latest/&sa=D&source=editors&ust=1637991494012000&usg=AOvVaw1Hf4ClTS50sD3RlYv5THzA), the most popular graph deep learning framework built on top of Pytorch. As we teach in the course, PyG is suitable to quickly implement GNN models, with abundant graph models already implemented. Moreover, GraphGym, available as part of the PyG, allows a training/evaluation pipeline to be built in a few lines of code.



We are going to publish the best tutorials on PyG.org website and promote them via different channels, such as Twitter!

# **Real-world application domains of graph ML**

Below, we list several real-world application domains, specifying how graphs and tasks are defined in these domains, as well as representative graph ML models and public datasets that can be used for your investigation. These are just examples; you should feel free to investigate other domains, models, and public datasets.

- ## **Recommender systems**

**Graphs**:

- Nodes: Users, items
- Edges: User-item interactions

**Tasks**:

- Predicting the edge ratings. Metric: RMSE
- Predicting edge existence. Metric: Recall@K

**Model(s)**: [LightGCN](https://www.google.com/url?q=https://arxiv.org/abs/2002.02126&sa=D&source=editors&ust=1637991494013000&usg=AOvVaw24Js_gXHSOBWfDeCNGzf9_)

**Public datasets**: [Movielens](https://www.google.com/url?q=https://grouplens.org/datasets/movielens/&sa=D&source=editors&ust=1637991494014000&usg=AOvVaw26ndsvoVZRYSw9V3lpdDeb), [Recsys repository](https://www.google.com/url?q=https://cseweb.ucsd.edu/~jmcauley/datasets.html&sa=D&source=editors&ust=1637991494014000&usg=AOvVaw1zUAV9sskz2Fd0rkTO_-VF)

- ## **Molecule classification**

**Graphs:**

- Nodes: Atoms
- Edges: Bonds

**Tasks:**

- Predicting properties of molecules. Metric: ROC-AUC

**Model(s):** See the [leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_graphprop/%23ogbg-molhiv&sa=D&source=editors&ust=1637991494015000&usg=AOvVaw2P6cvHhGeSM4YbdUmK18us)

**Datasets:** [ogbl-molhiv](https://www.google.com/url?q=https://ogb.stanford.edu/docs/graphprop/%23ogbg-mol&sa=D&source=editors&ust=1637991494015000&usg=AOvVaw2f9I-juCZkXI_GfWWFod_C)

- ## **Paper citation graphs**

**Graphs:**

- Nodes: Papers
- Edges: Paper citations

**Tasks:**

- Predicting subject areas of papers. Metric: Classification accuracy

**Model(s):** See the [leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_nodeprop/%23ogbn-arxiv&sa=D&source=editors&ust=1637991494016000&usg=AOvVaw1d5Ua-e4Ht-hTtotx0O3rK)

**Datasets:** [ogbn-arxiv](https://www.google.com/url?q=https://ogb.stanford.edu/docs/nodeprop/%23ogbn-arxiv&sa=D&source=editors&ust=1637991494016000&usg=AOvVaw1yiKLnOJZGJ3n1LZ_xpuTd)

- ## **Knowledge graph**

**Graphs:**

- Nodes: Entities
- Edges: Knowledge triples

**Tasks:**

- Predicting missing triples. Metric: Mean Reciprocal Rank (MRR)

**Model(s):** [TransE](https://www.google.com/url?q=https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html&sa=D&source=editors&ust=1637991494017000&usg=AOvVaw3M5Jk98uXoyr-jezIcYAQG), [DistMult](https://www.google.com/url?q=https://arxiv.org/abs/1412.6575&sa=D&source=editors&ust=1637991494017000&usg=AOvVaw2vPIpUVG5lXGoJnsl9a9ii), [ComplEx](https://www.google.com/url?q=https://arxiv.org/abs/1606.06357&sa=D&source=editors&ust=1637991494018000&usg=AOvVaw2yzxEDcHjOoN1z_ToLQ03T), [RotatE](https://www.google.com/url?q=https://arxiv.org/abs/1902.10197&sa=D&source=editors&ust=1637991494018000&usg=AOvVaw0-8p5JuUa9aAWHrZS76siU)

**Datasets:** [FB15k-237](https://www.google.com/url?q=https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/FB15k-237&sa=D&source=editors&ust=1637991494018000&usg=AOvVaw1PhZKBPKmwdOmnqq7y3mH-), [WN18RR](https://www.google.com/url?q=https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/wn18rr&sa=D&source=editors&ust=1637991494018000&usg=AOvVaw02IHcjvXIwzrVPYd5yphRl)

- ## **Author collaboration networks**

**Graphs:**

- Nodes: Authors
- Edges: Author collaboration

**Tasks:**

- Predicting future author collaboration. Metric: Hits@50

**Example model(s):** See the [leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_linkprop/%23ogbl-collab&sa=D&source=editors&ust=1637991494019000&usg=AOvVaw3h8s6phtWfH6L4bFm33YRt)

**Datasets:** [ogbl-collab](https://www.google.com/url?q=https://ogb.stanford.edu/docs/linkprop/%23ogbl-collab&sa=D&source=editors&ust=1637991494020000&usg=AOvVaw0lXwc9TXHSh3yUilWfKd8r)

- ## **Heterogeneous academic graph**

**Graphs:**

- Nodes: Papers, authors, institutions, fields of study
- Edges: an author is “affiliated with” an institution, an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic of” a field of study

**Tasks:**

- Predicting paper publication venues. Metric: Classification accuracy

**Example model(s):** See the [leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_nodeprop/%23ogbn-mag&sa=D&source=editors&ust=1637991494021000&usg=AOvVaw0QduXa3lQkm-_yuLxJaXt0).

**Datasets:** [ogbn-mag](https://www.google.com/url?q=https://ogb.stanford.edu/docs/nodeprop/%23ogbn-mag&sa=D&source=editors&ust=1637991494021000&usg=AOvVaw1hMwzDUYHKSOWxCbV9Ph9R)

**Note**: This is a medium-scale OGB dataset that requires more computational resources than the other datasets.

- ## **Product co-purchasing graph**

**Graphs:**

- Nodes: Products
- Edges: Product co-purchasing relations

**Tasks:**

- Predicting product categories. Metric: Classification accuracy

**Example model(s):** See the [leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_nodeprop/%23ogbn-products&sa=D&source=editors&ust=1637991494022000&usg=AOvVaw08q14TTU4V_3fZS1e5ifVK)

**Datasets:** [ogbn-products](https://www.google.com/url?q=https://ogb.stanford.edu/docs/nodeprop/%23ogbn-products&sa=D&source=editors&ust=1637991494022000&usg=AOvVaw3LWiwu3e9zxq1iSjJlX01K)

**Note**: This is a medium-scale OGB dataset that requires more computational resources than the other datasets.

- ## **Fraud Detection in Transaction Graphs**

**Graphs:**

- Nodes: Financial users (customers, banks)
- Edges: Transaction (money and amount sent)

**Tasks:**

- Edge classification - predict which edges are fraudulent. Metric: Hits@50

**Example model(s):** See [https://github.com/safe-graph/graph-fraud-detection-papers](https://www.google.com/url?q=https://github.com/safe-graph/graph-fraud-detection-papers&sa=D&source=editors&ust=1637991494023000&usg=AOvVaw2Q_MJTCFT5a1qbeDpDp4U3) 

**Datasets:** [Bitcoin Fraud Dataset (only use labeled data!)](https://www.google.com/url?q=https://www.kaggle.com/ellipticco/elliptic-data-set&sa=D&source=editors&ust=1637991494024000&usg=AOvVaw0Gd863UMJxxgxx3zfhz1OX)

- ## **Protein-Protein Interaction Networks**

**Graphs:**

- Nodes: Gene nodes
- Edges: Interaction between gene nodes

**Tasks:**

- Node classification - protein function prediction. Metric: Classification accuracy

**Example model(s):** See methods on [OGB node classification leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_nodeprop/&sa=D&source=editors&ust=1637991494025000&usg=AOvVaw1zdMKl5G7CECXq6Ynp48SU)

**Datasets:** [https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html](https://www.google.com/url?q=https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html&sa=D&source=editors&ust=1637991494025000&usg=AOvVaw0ZgfnKzCThUbABIfXmWWlU)

- ## **Drug-Drug Interaction Network**

**Graphs:**

- Nodes: FDA-approved or experimental drug
- Edges: interactions between drugs (joint effect of taking the two drugs together)

**Tasks:**

- predict drug-drug interactions - Metric: Hits@K

**Example model(s):** See the [leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_nodeprop/%23ogbn-products&sa=D&source=editors&ust=1637991494026000&usg=AOvVaw1z6_hMmMp-d_fqJc9wJFhl)

**Datasets:** [ogbl-ddi](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_linkprop/%23ogbl-ddi&sa=D&source=editors&ust=1637991494026000&usg=AOvVaw3Wzlm3mOT6UdRFrJ14CYV4)

- ## **Friend recommendation**

**Graphs:** social network

- Nodes: users
- Edges: potentially heterogeneous -- friend, follow, reply to, message, like, etc.

**Tasks:**

- Recommending/ranking new friends for user -- metrics: Hits@K, NDCG@K, MRR

**Example model(s):** GraFRank ([paper](https://www.google.com/url?q=http://nshah.net/publications/GrafRank.WWW.21.pdf&sa=D&source=editors&ust=1637991494027000&usg=AOvVaw2BoVmjLClgHeAPRP6_snnU), [GitHub](https://www.google.com/url?q=https://github.com/aravindsankar28/GraFRank&sa=D&source=editors&ust=1637991494028000&usg=AOvVaw0G5L5US-_Y7qfSFqC-ALg2))

**Datasets:** [Facebook](https://www.google.com/url?q=http://snap.stanford.edu/data/ego-Facebook.html&sa=D&source=editors&ust=1637991494028000&usg=AOvVaw1xpkX9erT1uMv00sTIeUKg), [Google+](https://www.google.com/url?q=http://snap.stanford.edu/data/ego-Gplus.html&sa=D&source=editors&ust=1637991494028000&usg=AOvVaw1py7qZUct2ooqvenBYx7Dn), [Twitter](https://www.google.com/url?q=http://snap.stanford.edu/data/ego-Twitter.html&sa=D&source=editors&ust=1637991494028000&usg=AOvVaw3To_zvufWx5EiIKX60s0-w)







You are free to choose any of the above domains and find other public datasets for the application that you choose. You can adopt the readily available graph datasets from [Open Graph Benchmark](https://www.google.com/url?q=https://ogb.stanford.edu/&sa=D&source=editors&ust=1637991494029000&usg=AOvVaw1FGjOE477gChihLTPxMy3F) (OGB)***** and the datasets available in [PyG](https://www.google.com/url?q=https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html&sa=D&source=editors&ust=1637991494029000&usg=AOvVaw3LtVDh8MNCTZaHEojlZ0pT). Alternatively, you are free to explore other application domains and the corresponding public datasets yourself.



*****Note that some OGB datasets are quite large and require a decent amount of computational resources to handle. We recommend using small-scale OGB datasets that are more tractable.

# **Resource on graph ML**

State-of-the-art models for graph ML can be found at

- [OGB leaderboard](https://www.google.com/url?q=https://ogb.stanford.edu/docs/leader_overview/&sa=D&source=editors&ust=1637991494030000&usg=AOvVaw1bn2CrjAcejnL-kKwE5o2O)
- Top ML conferences

- To find papers related to graph ML, you can search for the term “graph” across the titles.

- ICML 2019: [http://proceedings.mlr.press/v97/](https://www.google.com/url?q=http://proceedings.mlr.press/v97/&sa=D&source=editors&ust=1637991494030000&usg=AOvVaw0Lse06jUixl6kPrUfsE307)
- ICML 2020: [http://proceedings.mlr.press/v119/](https://www.google.com/url?q=http://proceedings.mlr.press/v119/&sa=D&source=editors&ust=1637991494031000&usg=AOvVaw25rtWkXbKTkfFgwiIM1Roe)
- ICML 2021: https://proceedings.mlr.press/v139/
- NeurIPS 2019: [https://papers.nips.cc/paper/2019](https://www.google.com/url?q=https://papers.nips.cc/paper/2019&sa=D&source=editors&ust=1637991494031000&usg=AOvVaw2hr7LzK9yNmANcSaEl8YZ6)
- NeurIPS 2020: [https://papers.nips.cc/paper/2020](https://www.google.com/url?q=https://papers.nips.cc/paper/2020&sa=D&source=editors&ust=1637991494031000&usg=AOvVaw3FFl33VwB6ZBuuLgu_Np7c)
- ICLR 2019: https://openreview.net/group?id=ICLR.cc/2019/Conference
- ICLR 2020: [https://openreview.net/group?id=ICLR.cc/2020/Conference](https://www.google.com/url?q=https://openreview.net/group?id%3DICLR.cc/2020/Conference&sa=D&source=editors&ust=1637991494032000&usg=AOvVaw2doN2RPhBQgP8J_B_VrNQC)
- ICLR 2021: [https://openreview.net/group?id=ICLR.cc/2020/Conference](https://www.google.com/url?q=https://openreview.net/group?id%3DICLR.cc/2020/Conference&sa=D&source=editors&ust=1637991494032000&usg=AOvVaw2doN2RPhBQgP8J_B_VrNQC)
- KDD 2019: [https://www.kdd.org/kdd2019/proceedings/](https://www.google.com/url?q=https://www.kdd.org/kdd2019/proceedings/&sa=D&source=editors&ust=1637991494032000&usg=AOvVaw1dUPmjSMJY6dAB-YVcreyU) 
- KDD 2020: [https://www.kdd.org/kdd2020/proceedings/](https://www.google.com/url?q=https://www.kdd.org/kdd2020/proceedings/&sa=D&source=editors&ust=1637991494032000&usg=AOvVaw3oDvL7cKO9jA4-4N2Ft6d4)
- KDD 2021: [https://www.kdd.org/kdd2021/proceedings/](https://www.google.com/url?q=https://www.kdd.org/kdd2020/proceedings/&sa=D&source=editors&ust=1637991494033000&usg=AOvVaw1wVwpwJ_s1lnTa8CP71fBS)

# **A Tutorial on Writing Blog Posts**

First, please read [this article](https://www.google.com/url?q=https://towardsdatascience.com/questions-96667b06af5&sa=D&source=editors&ust=1637991494033000&usg=AOvVaw2IDHihU0Ge9PVgiN8t_JuS) carefully to learn about how to write machine learning blog posts. For this course project, please also follow the instructions below.



**In the blog posts, you should include the** **following****:**

- At the beginning of your blog post, include “By XXX, YYY, ZZZ as part of the Stanford CS224W course project.”, where XXX, YYY, ZZZ are the names of the team members.
- The domain that you are applying graph ML to.
- Dataset descriptions (source, pre-processing etc).
- Step-by-step explanation of graph ML techniques you are applying.

- You can assume the following for the readers.

- Readers are familiar with machine learning (e.g., [CS229](https://www.google.com/url?q=https://cs229.stanford.edu/&sa=D&source=editors&ust=1637991494034000&usg=AOvVaw0XjG5UjGk1tziTyZ0DOHJD)) and deep learning (e.g., [CS230](https://www.google.com/url?q=https://cs230.stanford.edu/&sa=D&source=editors&ust=1637991494034000&usg=AOvVaw2f1m6L4HO0wuk1XGHFbCsN)) concepts. You do not need to explain them in detail.
- Readers are familiar with Pytorch.
- Readers are not familiar with graph ML (taught in [this course](https://www.google.com/url?q=https://web.stanford.edu/class/cs224w/&sa=D&source=editors&ust=1637991494035000&usg=AOvVaw1c7LjZTdOGP2kySTbJ7JuS).)

- **Visualizations** that would make the blog posts intriguing to read.

- Gifs > Images > Text to show your methods and results
- Try to use videos, images, flow charts as much as possible
- The more visualization, the better. Reading text-occupied blogs is often painful.

- Some code snippets of how you used PyG/Pytorch to implement the techniques
- Results that you obtain using the model on the dataset
- Image credits if you are adopting figures from other places (we encourage you to make your own figures).
- Link to your Google Colab that can be used to reproduce your results.
- Avoid criticizing research / research orgs. You are here to showcase your work, not to write opinion pieces.



**A good blog post should**

- be fun to read with many figures and visualizations.

- be easy to follow even for graph ML novice.
- clearly convey the potential of graph ML to the application domain.
- contain a good amount of PyG code snippets to understand how PyG is utilized to build the model.



**Using Medium**

For this course project, please use [Medium](https://www.google.com/url?q=https://medium.com/&sa=D&source=editors&ust=1637991494036000&usg=AOvVaw0mIeCX-GyUGfIoa6ZQFhJM) to write your blog post. Writing a blog post in Medium is super easy and intuitive. Below are the step-by-step instructions

- [Sign up / sign in on Medium.](https://www.google.com/url?q=https://medium.com/m/signin&sa=D&source=editors&ust=1637991494037000&usg=AOvVaw1kR-TPsr6G607Vm1gtE-ZG)
- For each group, one member should set up a draft

- To start a new draft, go [here](https://www.google.com/url?q=https://medium.com/new-story&sa=D&source=editors&ust=1637991494037000&usg=AOvVaw2aSUKeDqqMFF94aieWKYdI).
- To restart from the existing draft, go [here](https://www.google.com/url?q=https://medium.com/me/stories/drafts&sa=D&source=editors&ust=1637991494037000&usg=AOvVaw210ifwQ5ZqKW8ycr6g9oD1).

- Editing the draft

- Only one member (who owns the draft) can directly edit the blog post.
- If you want to write the blog together with your members, we suggest you work on the Google doc first and then copy-paste the eventual texts/images to the Medium draft.

- Submitting your draft

- The final product of the course project will be the **draft link** of your blog post.
- On the editing page, click the “...” button (located right next to the “publish”).
- Then click “Share draft link” to get the link.
- Please also share your account name with us.

- We will add you as a writer to the [stanford-cs224w publication page](https://www.google.com/url?q=https://medium.com/stanford-cs224w&sa=D&source=editors&ust=1637991494038000&usg=AOvVaw1Hs_PF0ZAqrjCr0zKdnw0t).
- Once you are added as our writer, you can click “...” button and further click “Add to publication”.

- **Do NOT publish your blog without our review.**

- TA reviewing & grading

- Once you have submitted your draft link as your final course project, TAs will then review and grade your blog posts.
- For those blogs that are easy-to-follow and technically sound, we would like to publish your blogs (under your accounts). We will also advertise your blogs under PyG.org and through our social media.
- You will get grading regardless of whether your blogs are published or not.





**Examples of good blog posts:**

- [http://peterbloem.nl/blog/transformers](https://www.google.com/url?q=http://peterbloem.nl/blog/transformers&sa=D&source=editors&ust=1637991494039000&usg=AOvVaw2CyqH6URIuhtB_-hlkUMuk) 
- [https://tkipf.github.io/graph-convolutional-networks/](https://www.google.com/url?q=https://tkipf.github.io/graph-convolutional-networks/&sa=D&source=editors&ust=1637991494040000&usg=AOvVaw22e2lMeUwM_ejcDDM6KXed)
- [https://towardsdatascience.com/graph-neural-networks-as-neural-diffusion-pdes-8571b8c0c774](https://www.google.com/url?q=https://towardsdatascience.com/graph-neural-networks-as-neural-diffusion-pdes-8571b8c0c774&sa=D&source=editors&ust=1637991494040000&usg=AOvVaw0Wsq8cNNVF0OZUudtnSdtX)
- [https://blog.twitter.com/engineering/en_us/topics/insights/2021/temporal-graph-networks](https://www.google.com/url?q=https://blog.twitter.com/engineering/en_us/topics/insights/2021/temporal-graph-networks&sa=D&source=editors&ust=1637991494040000&usg=AOvVaw1LwlY95jdkFMZAPR5fe3Fh)

# **Instructions on Google Colab**

Google Colab should include:

- a high-level summary of what the code is about and what the task is
- all the code to reproduce your results in the blog posts (including data preprocessing, model definition, and train/evaluation pipeline).
- detailed comments of what each cell does.

Examples of good Google colabs can be found at

- [https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html](https://www.google.com/url?q=https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html&sa=D&source=editors&ust=1637991494041000&usg=AOvVaw3V2iRScRoFlPtUz1WWzN_c)



# **Evaluation criteria (20% of the total course grade)**

## **Project proposal (20% of the project grade)** **[By October 19, 11:59 pm PST]**

- Application domains (10% / 20%)

- Which dataset are you planning to use?
- Describe the dataset/task/metric.
- Why did you choose the dataset?

- Graph ML techniques that you want to apply (10% / 20%)

- Which graph ML model are you planning to use?
- Describe the model (try using figures and equations).
- Why is the model appropriate for the dataset you have chosen?

- Note: It is ok to change the domains and techniques in your final proposal. The main purpose of the proposal is for you to get started early and for us to give you early feedback.



**Format:** Please use the Latex file of the NeurIPS conference: [https://nips.cc/Conferences/2020/PaperInformation/StyleFiles](https://www.google.com/url?q=https://nips.cc/Conferences/2020/PaperInformation/StyleFiles&sa=D&source=editors&ust=1637991494043000&usg=AOvVaw0pN1Q06xy95zeaVittVpB0)

Please aim for 2 pages without references. The abstract is not needed.

## **Final project (80% of the project grade)** **[[By December 9, 11:59 pm PST\]](https://www.google.com/url?q=https://docs.google.com/document/d/12bkLcJEAon9HdOKtyfywf3-kbfofz1HMBJZJTBR2_SY/edit%23heading%3Dh.3cacl4i2bqay&sa=D&source=editors&ust=1637991494043000&usg=AOvVaw0rhUiP6TylutTWL9BxUNG2)**

- Blog posts (60% / 80%)

- Blog structure and organization (20% / 60%)
- Effective use of figures/gifs to explain the concepts/techniques (20% / 60%)
- Technical soundness (20% / 60%)

- Google Colab (20% / 80%)

- Code correctness (10% / 20%)
- Documentation (10% / 20%)

# **Extra credit for PyG (up to 3% of total course grade)**

- You will also be considered for up to an extra 1-3% of the total grade if you create an approved pull requests to OGB ([https://github.com/snap-stanford/ogb](https://www.google.com/url?q=https://github.com/snap-stanford/ogb&sa=D&source=editors&ust=1637991494045000&usg=AOvVaw1ilB2vc9MdAZHe_vgLyeM4)),  PyG ([https://github.com/pyg-team/pytorch_geometric](https://www.google.com/url?q=https://github.com/pyg-team/pytorch_geometric&sa=D&source=editors&ust=1637991494045000&usg=AOvVaw29sIYk9ZIz0-nrKnX3X3Wp)), GraphGym ([https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/graphgym](https://www.google.com/url?q=https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/graphgym&sa=D&source=editors&ust=1637991494045000&usg=AOvVaw2OpIpF3eyTdhd9WTCyYzYT)) that add useful functionalities or fix bugs, depending on the significance of the contribution.
- You are highly encouraged to participate in the open-source development of our own in-house GNN platform PyG, such as

- finding bugs and creating issues
- finding interesting methods/papers to implement, creating a feature request, discussing the overall interface and implementation, and implementing the final proposal
- proposing missing functionality and implementing it
- writing Google Colabs/tutorials to show-case specific GNN pipelines/architectures/training protocols via PyG
- contributing useful modules to GraphGym pipeline ([https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym/custom_graphgym](https://www.google.com/url?q=https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym/custom_graphgym&sa=D&source=editors&ust=1637991494046000&usg=AOvVaw0nbKX5UpnnxqtUEHf8d6c2))
- contributing your final project to the OGB leaderboard together with open-source code using PyG. This is particularly exciting if your model is able to achieve SoTA performance.

- We highly recommend consulting the TAs or PyG lead developers (Matthias Fey, matthias.fey@tu-dortmund.de and Jiaxuan You, jiaxuan@cs.stanford.edu) before contributions so that we can confirm the details of incorporating the contribution and granting the extra grade.