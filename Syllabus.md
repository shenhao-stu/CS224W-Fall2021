# CS224W: Machine Learning with Graphs

### Stanford / Fall 2021

## Logistics

- **Lectures:** are on Tuesday/Thursday 1:30-3pm **in person** in the [NVIDIA Auditorium](https://campus-map.stanford.edu/?srch=NVIDIA+Auditorium).
- **Lecture Videos:** are available on [Canvas](https://canvas.stanford.edu/courses/144017/external_tools/3367) for all the enrolled Stanford students.
- **Public resources**: The lecture slides and assignments will be posted online as the course progresses. We are happy for anyone to use these resources, but we cannot grade the work of any students who are not officially enrolled in the class.
- **Contact**: Students should ask *all* course-related questions on Ed (accessible from Canvas), where you will also find announcements. For external inquiries, personal matters, or in emergencies, you can email us at *cs224w-aut2122-staff@lists.stanford.edu*.
- **Academic accommodations**: If you need an academic accommodation based on a disability, you should initiate the request with the [Office of Accessible Education (OAE)](https://oae.stanford.edu/accommodations/academic-accommodations). The OAE will evaluate the request, recommend accommodations, and prepare a letter for the teaching staff. Once you receive the letter, send it to our staff email address. Students should contact the OAE as soon as possible since timely notice is needed to coordinate accommodations.

## Content

###  What is this course about?

Complex data can be represented as a graph of relationships between objects. Such networks are a fundamental tool for modeling social, technological, and biological systems. This course focuses on the computational, algorithmic, and modeling challenges specific to the analysis of massive graphs. By means of studying the underlying graph structure and its features, students are introduced to machine learning techniques and data mining tools apt to reveal insights on a variety of networks.
**Topics include:** representation learning and Graph Neural Networks; algorithms for the World Wide Web; reasoning over Knowledge Graphs; influence maximization; disease outbreak detection, social network analysis.

###  Previous Offerings

You can access slides and project reports of previous versions of the course on our archived websites: [CS224W: Winter 2021](http://snap.stanford.edu/class/cs224w-2020) / [CS224W: Fall 2019](http://snap.stanford.edu/class/cs224w-2019) / [CS224W: Fall 2018](http://snap.stanford.edu/class/cs224w-2018) / [CS224W: Fall 2017](http://snap.stanford.edu/class/cs224w-2017) / [CS224W: Fall 2016](http://snap.stanford.edu/class/cs224w-2016) / [CS224W: Fall 2015](http://snap.stanford.edu/class/cs224w-2015) / [CS224W: Fall 2014](http://snap.stanford.edu/class/cs224w-2014) / [CS224W: Fall 2013](http://snap.stanford.edu/class/cs224w-2013) / [CS224W: Fall 2012](http://snap.stanford.edu/class/cs224w-2012) / [CS224W: Fall 2011](http://snap.stanford.edu/class/cs224w-2011) / [CS224W: Fall 2010](http://snap.stanford.edu/class/cs224w-2010)

###  Prerequisites

Students are expected to have the following background:

- Knowledge of basic computer science principles, sufficient to write a reasonably non-trivial computer program (e.g., CS107 or CS145 or equivalent are recommended)
- Familiarity with the basic probability theory (CS109 or Stat116 are sufficient but not necessary)
- Familiarity with the basic linear algebra (any one of Math 51, Math 103, Math 113, or CS 205 would be much more than necessary)

The recitation sessions in the first weeks of the class will give an overview of the expected background.

###  Course Materials

Notes and reading assignments will be posted periodically on the course Web site. The following books are recommended as optional reading:

- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) by William L. Hamilton
- [Networks, Crowds, and Markets: Reasoning About a Highly Connected World](http://www.cs.cornell.edu/home/kleinber/networks-book/) by David Easley and Jon Kleinberg
- [Network Science](http://networksciencebook.com/) by Albert-László Barabási

## Schedule

Lecture slides will be posted here shortly before each lecture.

| Date       | Description                                                  | Suggested Readings / Important Notes | Events                                                       | Deadlines                                                    |
| :--------- | :----------------------------------------------------------- | :----------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Tue Sep 21 | 1. Introduction; Machine Learning for Graphs [[slides](http://web.stanford.edu/class/cs224w/slides/01-intro.pdf)] |                                      |                                                              |                                                              |
| Thu Sep 23 | 2. Traditional Methods for ML on Graphs [[slides](http://web.stanford.edu/class/cs224w/slides/02-tradition-ml.pdf)] |                                      | [Colab 0](https://colab.research.google.com/drive/16tqEHKOLUgYvXKx1V3blfYGpQb1_09MG?usp=sharing), [Colab 1](https://colab.research.google.com/drive/1p2s0on6nibUYhJnONBWEAwpBlue37Tcc?usp=sharing) **out** |                                                              |
| Tue Sep 28 | 3. Node Embeddings [[slides](http://web.stanford.edu/class/cs224w/slides/03-nodeemb.pdf)] |                                      |                                                              |                                                              |
| Thu Sep 30 | 4. Link Analysis: PageRank [[slides](http://web.stanford.edu/class/cs224w/slides/04-pagerank.pdf)] |                                      | [Homework 1 ](http://web.stanford.edu/class/cs224w/homework/cs224w_hw1_fall_updated.pdf)**out** |                                                              |
| Tue Oct 5  | 5. Label Propagation for Node Classification [[slides](http://web.stanford.edu/class/cs224w/slides/05-message.pdf)] |                                      |                                                              |                                                              |
| Thu Oct 7  | 6. Graph Neural Networks 1: GNN Model [[slides](http://web.stanford.edu/class/cs224w/slides/06-GNN1.pdf)] |                                      | [Colab 2](https://colab.research.google.com/drive/1BRPw3WQjP8ANSFz-4Z1ldtNt9g7zm-bv?usp=sharing) **out** | Colab 1 **due**                                              |
| Tue Oct 12 | 7. Graph Neural Networks 2: Design Space [[slides](http://web.stanford.edu/class/cs224w/slides/07-GNN2.pdf)] |                                      |                                                              |                                                              |
| Thu Oct 14 | 8. Applications of Graph Neural Networks [[slides](http://web.stanford.edu/class/cs224w/slides/08-GNN-application.pdf)] |                                      | [Homework 2 ](http://web.stanford.edu/class/cs224w/homework/cs224w_hw2_fall_updated.pdf)**out** [LaTeX template](https://drive.google.com/file/d/10nfgROfq_4xH10sd5oCuT8P4peTaNnHJ/view?usp=sharing) | Homework 1 **due**                                           |
| Tue Oct 19 | 9. Theory of Graph Neural Networks [[slides](http://web.stanford.edu/class/cs224w/slides/09-theory.pdf)] |                                      |                                                              | Project Proposal** due**                                     |
| Thu Oct 21 | 10. Knowledge Graph Embeddings [[slides](http://web.stanford.edu/class/cs224w/slides/10-kg.pdf)] |                                      | [Colab 3](https://colab.research.google.com/drive/1bAvutxJhjMyNsbzlLuQybzn_DXM63CuE) **out** | Colab 2 **due**                                              |
| Tue Oct 26 | 11. Reasoning over Knowledge Graphs [[slides](http://web.stanford.edu/class/cs224w/slides/11-reasoning.pdf)] |                                      |                                                              |                                                              |
| Thu Oct 28 | 12. Frequent Subgraph Mining with GNNs [[slides](http://web.stanford.edu/class/cs224w/slides/12-motifs.pdf)] |                                      | [Homework 3](http://web.stanford.edu/class/cs224w/homework/cs224w_hw3_fall_updated.pdf) **out** [LaTeX template](http://web.stanford.edu/class/cs224w/homework/main.tex) | Homework 2 **due**                                           |
| Tue Nov 2  | **NO CLASS - DEMOCRACY DAY**                                 |                                      |                                                              |                                                              |
| Thu Nov 4  | 13. GNNs for Recommender Systems [[slides](http://web.stanford.edu/class/cs224w/slides/13-recsys.pdf)] |                                      | [Colab 4](https://colab.research.google.com/drive/1X4uOWv_xkefDu_h-pbJg-fEkMfR7NGz9?usp=sharing) **out** | Colab 3 **due**                                              |
| Tue Nov 9  | 14. Community Structure in Networks [[slides](http://web.stanford.edu/class/cs224w/slides/14-communities.pdf)] |                                      |                                                              |                                                              |
| Thu Nov 11 | 15. Deep Generative Models for Graphs [[slides](http://web.stanford.edu/class/cs224w/slides/15-deep-generation.pdf)] |                                      | [Colab 5](https://colab.research.google.com/drive/17Pe4o_oSsD2J-wTb_xGtYJQsyCawK6sJ?usp=sharing) **out** | Homework 3 **due**                                           |
| Tue Nov 16 | 16. Advanced Topics on GNNs [[slides](http://web.stanford.edu/class/cs224w/slides/16-advanced.pdf)] |                                      |                                                              |                                                              |
| Thu Nov 18 | 17. Scaling Up GNNs [[slides](http://web.stanford.edu/class/cs224w/slides/17-scalable.pdf)] |                                      |                                                              | Colab 4 **due**                                              |
| Fri Nov 19 |                                                              |                                      |                                                              | **Exam** [[Review slides](http://web.stanford.edu/class/cs224w/slides/Exam_Preparation.pdf)] |
| Tue Nov 23 | **NO CLASS - THANKSGIVING BREAK**                            |                                      |                                                              |                                                              |
| Tue Nov 25 | **NO CLASS - THANKSGIVING BREAK**                            |                                      |                                                              |                                                              |
| Tue Nov 30 | 18. **Guest Lecture**: TBD                                   |                                      |                                                              |                                                              |
| Thu Dec 2  | 19. GNNs for Science                                         |                                      |                                                              | Colab 5 **due**                                              |
| Thu Dec 9  |                                                              |                                      |                                                              | Project Report **due**                                       |