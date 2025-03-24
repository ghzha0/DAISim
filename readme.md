# The core code of DAISim


If this paper or code helps you, please cite:
```
@inproceedings{10.1145/3583780.3615060,
author = {Zhao, Guanhao and Huang, Zhenya and Zhuang, Yan and Liu, Jiayu and Liu, Qi and Liu, Zhiding and Wu, Jinze and Chen, Enhong},
title = {Simulating Student Interactions with Two-stage Imitation Learning for Intelligent Educational Systems},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615060},
doi = {10.1145/3583780.3615060},
abstract = {The fundamental task of intelligent educational systems is to offer adaptive learning services to students, such as exercise recommendations and computerized adaptive testing. However, optimizing required models in these systems would always encounter the collection difficulty of high-quality interaction data in practice. Therefore, establishing a student simulator is of great value since it can generate valid interactions to help optimize models. Existing advances have achieved success but generally suffer from exposure bias and overlook long-term intentions. To tackle these problems, we propose a novel Direct-Adversarial Imitation Student Simulator (DAISim) by formulating it as a Markov Decision Process (MDP), which unifies the workflow of the simulator in training and generating to alleviate the exposure bias and single-step optimization problems. To construct the intentions underlying the complex student interactions, we first propose a direct imitation strategy to mimic the interactions with a simple reward function. Then, we propose an adversarial imitation strategy to learn a rational distribution with the reward given by a parameterized discriminator. Furthermore, we optimize the discriminator in adversarial imitation in a pairwise manner, and the theoretical analysis shows that the pairwise discriminator would improve the generation quality. We conduct extensive experiments on real-world datasets, where the results demonstrate that our DAISim can simulate high-quality student interactions whose distribution is close to real distribution and can promote several downstream services.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {3423–3432},
numpages = {10},
keywords = {adaptive learning, imitation learning, student simulator},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```
