# Revisiting Topological Fairness in Graph Neural Networks from a Causal Perspective
## Abstract
Graph Neural Networks (GNNs) have significantly advanced various graph-based analysis tasks such as social network analysis and recommendation systems. However, as data-driven models, GNNs inherit biases from historical data, which raises fairness concerns. While most existing research focuses on fairness issues from the perspective of sensitive attributes, studies addressing fairness concerns from the topological structure perspective remain underdeveloped. Most related studies attribute topological bias to the homophily effect, where nodes with the same sensitive attribute are more likely to establish connections. Although the homophily effect is an important source of bias, it cannot encompass all types of topological bias, as the topological structure takes various forms, such as teacher-student relationships, family ties, 
or friendships on social media. These relationships are not always closely related to sensitive attributes, yet fairness bias is prevalent in almost all graph data. To explore the inherent bias within topological structure and uncover its unbiased nature, we propose the concept of Topological Fairness. Specifically, Topological Fairness categorizes neighborhood information into helpful information and harmful information, considering inappropriate neighborhood information as a confounding factor that causes bias. First, we conduct a theoretical analysis of this concept from a causal perspective. Subsequently, to differentiate between helpful and harmful information, we introduce a novel method named Graph Fairness Differentiation (GFD). By evaluating the sign of the derivative, we can effectively determine whether the information is helpful or harmful. Finally, our experiments on four datasets demonstrate the effectiveness of our approach, 
and it achieves competitive performance compared to other state-of-the-art models. Overall, the definition of topological fairness lays the foundation for future research, and the use of causal models to discuss topological bias has the potential to open up the scope for more research in this area.
![Overall](https://github.com/user-attachments/assets/3ea8a4ca-d0fc-49c4-8163-c2b84ffbf023)
## Implementation Details
The implementation is based on PyTorch. Specifically, we replicate each experiment using three different random seeds: 1, 10, and 100. All GNNs are optimized with Adam optimizer. The training is conducted for 1,000 epochs with learning rate chosen from \{0.1, 0.01, 0.001\}. The GNNs feature a hidden dimension of 16 and employ dropout rates selected from \{0.2, 0.5, 0.8\}. Additionally, we experiment with iteration numbers for Hessian matrix inverse estimation ranging from \{10, 50, 100, 500, 1000, 5000\} to optimize performance. All experiments are conducted on a device equipped with two GeForce RTX 4080 Super GPUs.
## Requirements
Python == 3.8.18
torch == 1.8.0 + cu111
cuda == 11.1
torch-geometric == 1.7.0
torch-scatter == 2.0.6
torch-sparse == 0.6.9
torch-spline-conv == 1.2.1
dgl == 0.6.1
scikit-learn == 0.23.1
numpy == 1.22.0
scipy == 1.4.1
networkx == 2.4
