# Revisiting Topological Fairness in Graph Neural Networks from a Causal Perspective
## Abstract
Graph Neural Networks (GNNs) have significantly advanced various graph-based analysis tasks such as social network analysis and recommendation systems. However, as data-driven models, GNNs inherit biases from historical data, raising fairness concerns. In graph data, the effect of topological structure on fairness has been an under-explored problem. Most existing studies attribute topological bias to the homophily effect of sensitive attributes, where nodes with the same sensitive attribute are more likely to establish connections. Here, we will rethink topological fairness from a fine-grained perspective and explore how different forms of links influence fairness. To achieve this, we first propose the concept of Topological Fairness and provide a theoretical analysis. Specifically, we categorize neighborhood information into appropriate and inappropriate, considering inappropriate neighborhood information as a confounding factor that introduces bias. Subsequently, we introduce a method named Graph Fairness Differentiation (GFD), which determines the appropriateness of information by evaluating the sign of the derivative. Finally, our experiments on four datasets achieve competitive performance compared to baseline methods, demonstrating the effectiveness of our approach.

## Introduction
Graph Representation Learning (GRL) is one of the fundamental problems in graph analysis, which has achieved successful applications in areas such as social network analysis, recommendation systems, and bioinformatics. The fundamental goal of GRL is to capture the semantic information of nodes and edges by learning their representation within graph structure. This process facilitates more efficient and precise analysis of graph data. Current methodologies in GRL range from matrix factorization \cite{1,2,3} and random walk methods \cite{4,5,6} to graph neural network techniques \cite{7,8,9}. Among these, Graph Neural Networks (GNNs) have demonstrated groundbreaking performance, standing out as the current hotspot and mainstream in GRL. Researchers have proposed numerous classical GNNs, including Graph Convolutional Network (GCN) \cite{7}, Graph Attention Networks (GAT) \cite{9}, and Graph Sampling and Aggregation (GraphSAGE) \cite{8}, among others. These models have become the primary framework for learning graph representations, which are utilized to address a diverse range of downstream tasks. However, due to the data-driven nature of GNNs, biases and inequalities may inevitably arise during model training \cite{10,11,12,13}. These biases primarily stem from unequal treatment in historical data, societal biases, sampling biases, etc. Such biases can have long-term effects on the model's training, decision-making, and prediction process, potentially leading to unfair decision for specific demographic groups.

To address fairness challenges, researchers have proposed a series of classic fairness concepts, primarily including Group Fairness \cite{14,15} and Individual Fairness \cite{16,17}. In the field of GNNs, significant progress has been made to address fairness issues, which can mainly be categorized into three types: adversarial debiasing, fairness constraints, and other innovative debiasing strategies \cite{12} (such as Bayesian-based methods and data preprocessing-based methods). However, how to rationally quantify and eliminate the effect of complex graph structures on fairness remains an urgent challenge. Currently, the homophily effect of sensitive attributes is widely regarded as the primary cause of topological bias, where nodes with the same sensitive attribute are more likely to form connections \cite{21,23}. Here, we aim to discuss topological bias from a finer-grained perspective, and explore how each different form of edge influences fairness, as shown in Figure \ref{framework1}.

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{Inspired.jpg}
  \caption{The figure illustrates our motivation: exploring how the disappearance of a specific edge influences the fairness of the model.}
  \label{framework1}
\end{figure}

To achieve this goal, we first introduce the concept of Topological Fairness and present a theoretical analysis. It is important to note that, unlike Structural Fairness \cite{26}, which primarily addresses long-tail effects, Topological Fairness focuses on exploring the unbiased nature of topological structures. Specifically, we categorize neighborhood information into appropriate information that benefits fairness predictions and inappropriate information that undermines fairness, where the latter is considered a confounding factor that amplifies bias. Subsequently, to effectively differentiate and intervene in inappropriate information, we introduce the Graph Fairness Differentiation (GFD) method. By achieving a minimum value of GFD within its domain, we can exclude all inappropriate information and identify the optimal fairness graph. The overall conceptual framework of our work is illustrated in Figure \ref{framework}. Finally, we conduct experiments on four datasets, and the results show competitive performance, validating the effectiveness of our proposed method. Our contributions can be summarized as follows:

\begin{itemize}
\item After rethinking topological bias from a fine-grained perspective, we introduce the concept of topological fairness and present a theoretical analysis.
\item To differentiate between appropriate and inappropriate information and to intervene in the latter, we propose the GFD method, which aims to eliminate the effects of confounding factors and to determine the optimal fairness graph.
\item We conduct experiments on four datasets, and the results demonstrate that our method achieves a competitive fairness-utility trade-off, validating the effectiveness of the proposed approach.
\end{itemize}

\begin{figure*}[h]
  \centering
  \includegraphics[width=\linewidth]{Overall.jpg}
  \caption{The figure illustrates the motivation behind GFD. For each node $v_i$, during the neighborhood information aggregation, our goal is to exclude all inappropriate information while incorporating all appropriate information. By minimizing $\nabla_{A_{ij}} \Gamma$, we can achieve this objective.}
  \label{framework}
\end{figure*}

\section{Related Work}
\subsection {Graph Neural Networks}

Graph Neural Networks encode attributes and structure into vector representations, effectively capturing useful graph structural information. Some well-known classical models, such as GCN \cite{7}, GAT \cite{9}, GraphSage \cite{8} and Graph Isomorphism Network (GIN) \cite{30}, have achieved significant success. The GCN aggregates features of neighboring nodes, extending the concept of traditional convolution to graph data through spectral methods. The GAT incorporates attention mechanisms to dynamically weigh the importance of adjacent nodes for feature aggregation. The GraphSAGE samples a fixed number of neighboring nodes and aggregates their features, suitable for large-scale graph data. The GIN learns graph isomorphism through a powerful aggregation function, aiming to elevate GNN capabilities to the level of the Weisfeiler-Lehman test. These methods are widely used for processing graph data, such as social networks, chemical structures, and transportation networks.

\subsection {Fairness in Graph Neural Networks}

Fairness is a multilevel connotation concept that encompasses various aspects, notably Group Fairness \cite{14,15}, Individual Fairness \cite{16,17}. Frontier research in the field, such as FairGNN \cite{10}, NIFTY \cite{11}, BIND \cite{13}, FairSIN \cite{25} has made significant progress, achieving debiasing while ensuring model efficiency and effectiveness. FairGNN employs an adversarial component to filter out sensitive attribute information from node embeddings, ensuring that the GNN classifier makes predictions independent of sensitive attribute, thereby achieving model debiasing. NIFTY simultaneously considers fairness and stability by introducing a novel objective function to optimize consistency between predictions made with perturbed sensitive attribute and their unperturbed counterparts.  BIND introduces a novel strategy for quantifying the influence of each training node on model bias and debiasing GNNs by removing nodes that are harmful to fairness. FairSIN proposes an innovative neutralization-based paradigm, where additional Fairness-facilitating Features are incorporated into node features or representations before message passing.

Some recent studies have empirically demonstrated that message passing process can exacerbates fairness bias \cite{25}. Consequently, addressing bias in graph structure has emerged as a novel and pressing challenge. However, this issue still lacks a formal definition, and thus we propose the concept of Topological Fairness. It is worth noting that, unlike the Structural Fairness \cite{26}, which primarily focuses on the issue of long-tail effects, Topological Fairness aims to explore the unbiased nature of topological structures. Recent methods, such as FairDrop \cite{21}, REFEREE \cite{31}, EDITS \cite{12}, Graphair \cite{32}, have made some progress. FairDrop proposes an edge masking algorithm to counter-act homophily and improve fairness. REFEREE provides structural explanations of topology bias and proposes a novel post-hoc explanation framework. EDITS focuses on input graph data, aiming to preprocess the data to enable fairer GNN outcomes by feeding the model with less biased information. Graphair considers both feature masking and topology modification, automatically discovers fairness-aware augmentations from input graphs. Although several studies have explored the fairness implications of graph topology structure, these methods have not considered the connectivity of edges, which means that removing one edge may result in changes to other edges. Therefore, we introduce the method of GFD to address this issue at the graph-level.

\section{Methodology}
\subsection {Preliminaries}
\subsubsection{Notations}

A graph is given by $\mathcal{G}=\{\mathcal{V},\mathcal{E}\}$, where $\mathcal{V}$ represents the set of nodes with $|\mathcal{V}|=n$ indicating the total number of nodes, $\mathcal{E}$ represents the set of edges with $|\mathcal{E}|=m$ indicating the total number of edges. $A\in \mathcal{R}^{|n|\times|n|}$ denote the adjacency matrix of the input graph $\mathcal{G}$, where $A_{ij}=1$ if there is an edge between nodes $v_i$ and $v_j$, and $A_{ij}=0$ otherwise. $X\in \mathcal{R}^{n\times d}$ denotes the node feature matrix, where $d$ is the node attribute dimension, $x_i\in \mathcal{R}^d$ is the feature vector for node $v_i\in \mathcal{V}$. Furthermore, $\mathcal{S}\in\{0,1\}^n$ represents the binary sensitive attribute of the nodes.

\subsubsection{Structural Causal Model}

The Structural Causal Model (SCM), conceived by Judea Pearl \cite{27}, is proposed for modeling causal relationships among variables. Central to SCM is the utilization of a directed graph, typically a Directed Acyclic Graph (DAG). In this graphical representation, nodes represent variables, while directed edge depict direct causal influences. Judea Pearl formally defined a causal model as an ordered triple $<U,V,F>$, where $U$ denotes a set of exogenous variables whose values are determined solely by external factors, $V$ signifies a set of endogenous variables whose values are determined by both internal factors and exogenous variables, and $F$ is a set of structural equations that describe how the values of specific endogenous variables are calculated based on the values of other variables.

\subsubsection{Fairness Metrics}

When assessing fairness, two pivotal concepts commonly introduced are Statistical Parity (also known as Demographic Parity) \cite{16} and Equal Opportunity \cite{34}. Statistical Parity primarily focuses on whether distribution of predictions is consistent across different sensitive subgroups. Specifically, it aims to ensure that, for each value of a sensitive attribute, the predictions maintain roughly equal proportions between positive and negative classes. Mathematically, Statistical Parity can be expressed as,
\begin{equation}
  P_{\hat{y}}^{(\mathcal{S}=0)} = P_{\hat{y}}^{(\mathcal{S}=1)}
\end{equation}
where $ P_{\hat{y}}^{(\mathcal{S}=0)} $ and $ P_{\hat{y}}^{(\mathcal{S}=1)} $ denote distribution of the probabilistic predictions in $ \hat{Y}^{(\mathcal{S}=0)} $ and $ \hat{Y}^{(\mathcal{S}=1)} $, respectively.

Equal Opportunity is another important fairness metric that is usually applied to binary classification problems. It concerns the model's ability to distinguish positive instances in different subgroups defined by sensitive attribute. Specifically, Equal Opportunity requires the model to maintain similar True Positive Rate (also known as sensitivity or recall) across different subgroups defined by sensitive attribute. This means that the model should correctly identify positive instances with similar probabilities regardless of the subgroup. In mathematical terms, Equal Opportunity can be formulated as,
\begin{equation}
  P_{\hat{y}}^{(S=0,Y=1)} = P_{\hat{y}}^{(S=1,Y=1)}
\end{equation}
where $ P_{\hat{y}}^{(\mathcal{S}=0,Y=1)} $ and $ P_{\hat{y}}^{(\mathcal{S}=1,Y=1)} $ are model prediction distributions for nodes with $ (\mathcal{S} = 0,Y = 1) $ and $ (\mathcal{S} = 1,Y = 1) $, respectively.

\subsubsection{Probabilistic Distribution Disparity.}

Probabilistic Distribution Disparity (PDD) is a bias quantification strategy proposed by \cite{13}, which can be instantiated with various fairness notions to depict the model bias from different perspectives. Specifically, PDD is defined as the Wasserstein-1 distance between the probability distributions of a variable of interest in different sensitive subgroups. For instance, the PDD instantiated with Statistical Parity $\Gamma_{SP}$ is,
\begin{equation}
  \Gamma_{SP} = Wasserstein_1(P_{\hat{y}}^{(\mathcal{S}=0)},P_{\hat{y}}^{(\mathcal{S}=1)})
\end{equation}
where $ Wasserstein_1(\cdot,\cdot) $ represents the Wasserstein-1 distance between two distributions.

Similarly, the PDD instantiated with Equal Opportunity $\Gamma_{EO}$ is,
\begin{equation}
  \Gamma_{EO} = Wasserstein_1(P_{\hat{y}}^{(\mathcal{S}=0,Y=1)},P_{\hat{y}}^{(\mathcal{S}=1,Y=1)})
\end{equation}

\subsection {Theoretical Analysis}
\subsubsection{Topological bias}

Topological bias is commonly attributed to the homophily effect, where nodes with the same sensitive attribute are more likely to establish connections \cite{21,23}. This phenomenon can be formalized as,
\begin{equation}
  P_{ij}(\mathcal{S}_i=\mathcal{S}_j)\gg P_{ij}(\mathcal{S}_i\neq \mathcal{S}_j)
\end{equation}
where $P_{ij}(\cdot)$ represents the probability of an edge between nodes $i$ and $j$, $\mathcal{S}_i=\mathcal{S}_j$ and $\mathcal{S}_i\neq \mathcal{S}_j$ denote nodes with the same and different sensitive attributes, respectively. Given that the information aggregation process can exacerbate fairness bias \cite{25}, the fairness issues inherent in complex graph structures have become increasingly urgent to address. A key question in this context is how topological structures influence fairness predictions.

\subsubsection{Topological Fairness}

To answer this question, we introduce the concept of Topological Fairness. The goal of Topological Fairness is to explore the unbiased nature of topological structures and to discover the potential invariant rationales \cite{35,36} within graphs. In this context, we categorize neighborhood information into appropriate and inappropriate information. Inappropriate information is regarded as a confounding factor that causes bias during the information aggregation process. GNNs can eliminate topological bias and achieve Topological Fairness if and only if all appropriate information is considered and inappropriate information is excluded.

\noindent \textbf{Definition 1. Topological Fairness.} Let $v_i$ be a node in a graph $\mathcal{G}$, $N(v_i)$ and $N'(v_i)$ denote the set of neighbors includes all neighborhood information and excludes inappropriate neighborhood information, respectively. For a GNN trained specifically for graph $\mathcal{G}$, whose aggregation process is $\Psi(v_i,\cdot)$, where $\cdot$ represents the set of neighborhood information utilized by the GNN, we define Topological Fairness is achieved if and only if the following equation is satisfied,
\begin{equation}
P(\hat{Y}_{v_i}=y|\Psi(v_i,N(v_i)))=P(\hat{Y}_{v_i}=y|\Psi(v_i,N'(v_i)))
\end{equation}

\subsubsection{Causal Analysis} 

We provide a theoretical analysis of our definition from a causal perspective. Specifically, we focus on the causal relationships among six variables: node features $X$, appropriate information $\overline{N}$, inappropriate information $\tilde{N}$, node embedding $E$, biased prediction $\tilde{Y}$ and exogenous variable $U$. We employ SCM to model the variables, where nodes represent abstract variables and directed edges denote the causal relationship between these variables. Our hypothetical SCM is based on the following assumptions: (1) $U$ represents all unobserved latent factors, and no variables outside the model influence the causal relationships. (2) The statistical independencies and dependencies among the variables in the causal graph are fully determined by the structure of the graph. (3) The causal relationships in the model are identifiable, meaning all causal paths and effects can be inferred from the observed data. With these assumptions, the SCM can be depicted in Figure \ref{causal}.

\begin{figure}[h]
  \centering
  \includegraphics[width=0.5\linewidth]{causal.jpg}
  \caption{SCM for the information aggregation process.}
  \label{causal}
\end{figure}

Our analysis centers on three causal paths:
\begin{itemize}
\item $\tilde{N} \rightarrow E \rightarrow \tilde{Y}$. Direct path: Inappropriate neighborhood information $\tilde{N}$ directly influences node embedding $E$, which in turn leads to the biased prediction $\tilde{Y}$.
\item $\tilde{N} \leftarrow U \rightarrow X \rightarrow E \rightarrow \tilde{Y}$ and $\tilde{N} \leftarrow U \rightarrow \overline{N} \rightarrow E \rightarrow \tilde{Y}$. Indirect path: Unobserved factors $U$ can indirectly cause biased prediction results.
\end{itemize}

\subsubsection {Graph Fairness Differentiation}

To effectively differentiate between appropriate and inappropriate neighborhood information in the given data and intervene in the inappropriate information, we propose the GFD method.

\noindent \textbf{Definition 2. Graph Fairness Differentiation.}  Let $\Gamma$ represent the PDD values and $A$ denote the adjacency matrix. Assuming $\Gamma$ is differentiable with respect to $A$, we define Graph Fairness Differentiation as $\nabla_{A_{ij}}\Gamma$.

The rationale behind this definition is that modifications to the topological structure of an input graph lead to changes in PDD values. If a topological change results in a higher PDD value, it indicates that the change leads to more unfair outcomes. This suggests that the modified topological structure contains more inappropriate neighborhood information. Therefore, by achieving the minimal value of $\nabla_{A_{ij}}\Gamma$ within its domain, we can effectively eliminate the influence of inappropriate neighborhood information. Specifically, to calculate the value of $\nabla_{A_{ij}}\Gamma$, we apply the chain rule and decompose the influence of the topology $A$ on the PDD value $\Gamma$ into two parts: the influence of topology $A$ on the optimal parameters $\hat{W}$, and the effect of these optimal parameters $\hat{W}$ on the PDD value $\Gamma$. It is important to note that PDD is a function of $\hat{W}$ for a trained GNN, and the optimal parameters $\hat{W}$ minimize the objective function in the node classification task \cite{13}. Mathematically, the optimal parameters $\hat{W}$ take the form of,
\begin{equation}
\hat{W} \triangleq \text{arg}\underset{W}{\text{min}} L_\mathcal{V}(G,W) = \text{arg}\underset{W}{\text{min}} L_\mathcal{V}(X,A,W)
\end{equation}

Subsequently, to determine how $W$ changes with alterations in $A$, we employ the method of implicit differentiation. The derivative of $\hat{W}$ with respect to $A$ is calculated as follows (see proofs in Appendix),
\begin{align}
&\frac{\partial \hat{W}}{\partial A_{ij}}=-(\frac{\partial^2 L_V(X,A,\hat{W})}{\partial W^2})^{-1} \notag
\\
&\cdot \frac{\frac{\partial L_V}{\partial W}(X,A_{ij}+\epsilon \cdot E_{ij},\hat{W})-\frac{\partial L_V}{\partial W}(X,A_{ij}-\epsilon \cdot E_{ij},\hat{W})}{2\epsilon}
\end{align}
where $E$ is a matrix of the same type as $W$, with all elements set to 0 except for the element at position $(i,j)$, which is set to 1. $\epsilon$ is a sufficiently small number, which means only minor increases and decreases of $\epsilon$ are applied to the elements in row $i$ and column $j$ of $W$. Consequently, $\nabla_{A_{ij}}\Gamma$ can be calculated as follows,
\begin{align}
&\nabla_{A_{ij}}\Gamma =(\frac{\partial \Gamma}{\partial W})^{\mathsf{T}} \cdot \frac{\partial \hat{W}}{\partial A_{ij}} =-(\frac{\partial \Gamma}{\partial W})^{\mathsf{T}} \cdot (\frac{\partial^2 L_\mathcal{V}(X,A,\hat{W})}{\partial W^2})^{-1} \notag
\\
&\cdot \frac{\frac{\partial L_\mathcal{V}}{\partial W}(X,A_{ij}+\epsilon \cdot E_{ij},\hat{W})-\frac{\partial L_\mathcal{V}}{\partial W}(X,A_{ij}-\epsilon \cdot E_{ij},\hat{W})}{2\epsilon}
\end{align}

The algorithmic routine for calculating $\nabla_{A_{ij}}\Gamma$ is provided in Algorithm \ref{Algorithm_1}. By achieving the minimal value of $\nabla_{A_{ij}}\Gamma$ within its domain, we can mitigate topological bias and identify the optimal fairness graph.

\begin{algorithm}[tb]
    \caption{Computation of $\nabla_{A_{ij}} \Gamma$}
    \label{Algorithm_1}
    \textbf{Input}: the processed graph data $\mathcal{G}$, the trained GNN model $f_{\hat{W}}$, the set of training nodes $\mathcal{V}$\\
    \textbf{Output}: $\nabla_{A_{ij}} \Gamma$
    \begin{algorithmic}[1] %[1] enables line numbers
        \STATE Initialize $\nabla_{A_{ij}} \Gamma$ = $\varnothing$.
        \STATE Compute {$\frac{\partial \Gamma}{\partial W}$ based on $f_{\hat{W}}$}.
        \STATE Compute {$\frac{\partial  L_\mathcal{V} (X, A, \hat{W})}{\partial W}$ based on $f_{\hat{W}}$}.
        \STATE Estimate the inverse of the Hessian matrix,
        \\
        $\left( \frac{\partial^2 L_\mathcal{V} (X, A, \hat{W})}{\partial W^2} \right)^{-1}$.
        \FOR{each element $A_{ij}$ in matrix $A$}
        \STATE Compute $\frac{\partial^2 L_\mathcal{V} (X, A, \hat{W})}{\partial W \partial A_{ij}}$.
        \ENDFOR
        \STATE Compute $\nabla_{A_{ij}} \Gamma$.
        \STATE Append the computed value to $\nabla_{A_{ij}} \Gamma$.
        \STATE \textbf{return} $\nabla_{A_{ij}} \Gamma$
    \end{algorithmic}
\end{algorithm}

\subsubsection{Complexity Analysis}

The computation of $\nabla_{A_{ij}}\Gamma$ primarily involves the calculation of the inverse of the Hessian matrix for the second term and the computation of the second derivatives for the third term. For the calculation of the inverse of the Hessian matrix, we refer to the work of \cite{13} and adopt the efficient estimation method proposed by \cite{41}. This widely recognized method has a time complexity linearly dependent on the number of parameters \cite{42,43}. Specifically, it estimates the product of the desired Hessian inverse and an arbitrary vector, where the vector has the same dimension as the columns of the Hessian matrix, thereby facilitating the computation of the multiplication between $\left( \frac{\partial^2 L_\mathcal{V} (X, A, \hat{W})}{\partial W^2} \right)^{-1}$ and $\frac{\frac{\partial L_\mathcal{V}}{\partial W} \left( X, A_{ij} + \epsilon \cdot E_{ij}, \hat{W} \right) - \frac{\partial L_\mathcal{V}}{\partial W} \left( X, A_{ij} - \epsilon \cdot E_{ij}, \hat{W} \right)}{2\epsilon}$. Assuming the number of parameters is $p$, the time complexity for computing the inverse of the Hessian matrix is $\mathcal{O}(p)$. Regarding the computation of the second-order derivatives, we first consider the time complexity of the first-order derivative $\frac{\partial L_\mathcal{V}}{\partial W}$. When calculating the first-order derivatives $\frac{\partial L_\mathcal{V}}{\partial W}$ for all training nodes, the time complexity will be $\mathcal{O}(np)$. To simplify this computation, in practice, for each partial derivative computation of $A_{ij}$, we only focus on the partial derivatives of its endpoints loss terms, ($\frac{\partial L_{v_1^{(i)}}}{\partial W}$ and $\frac{\partial L_{v_2^{(i)}}}{\partial W}$), each with a time complexity of $\mathcal{O}(p)$. Consequently, the time complexity for the second-order partial derivatives becomes $\mathcal{O}(n^2p)$. In summary, the total time complexity for computing $\nabla_{A_{ij}}\Gamma$ is $\mathcal{O}(n^2p^2)$.

\subsection {Implementations}

\subsubsection{Accuracy Loss} 

We employ the cross-entropy loss to pre-train the model, as it facilitates faster convergence during training and effectively handles the softmax output layer commonly used in classification tasks. Mathematically, the cross-entropy loss is expressed as,
\begin{equation}
  L_{CE}=-\sum_{i=1}^n [y_i \text{log}(\hat{y}_i)+(1-y_i) \text{log}(1-\hat{y}_i)]
\end{equation}
where $y_i$ represents the ground truth label and $\hat{y}_i$ represents the predicted probability for node $v_i$.

\subsubsection{Fairness Loss}

To improve the fairness performance of the model, we encourage GNNs to make fairer predictions by optimizing the topological structure. Specifically, for each input graph, we mitigate bias by minimizing $\nabla_{A_{ij}}\Gamma$ within its domain. The optimization problem can be formulated as follows,
\begin{equation}
  \text{arg}\underset{A_{ij}}{\text{min}} (\nabla_{A_{ij}}\Gamma + \lambda \parallel A_{ij} \parallel_F^2)
\end{equation}
where $\parallel \cdot \parallel_F^2$ represents the Frobenius norm, and $\lambda$ denotes the regularization parameter. Based on this optimization objective, the fairness loss function can be expressed as,
\begin{equation}
  L_{GFD}=\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \nabla_{A_{ij}}\Gamma
\end{equation}

\subsubsection{Graph Regularization Loss} 

To preserve the intrinsic geometric structure, ensuring that connected nodes remain close in the feature space, we introduce a graph regularization term \cite{37},
\begin{align}
  L_{REG}=-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n A_{ij} \parallel X_i - X_j \parallel_2^2 \notag
  =\text{tr}(XLX^{\mathsf{T}})
\end{align}
where $\parallel \cdot \parallel_2^2$ denotes the L2 norm. $\text{tr}(\cdot)$ denotes the trace of a matrix. $L=D-A$ represents the graph Laplacian matrix, $D$ is a diagonal matrix with diagonal elements equal to the row sums of $A$.

\subsubsection{Total loss}

By combining all of the above loss terms, the total loss function can be expressed as,
\begin{equation}
  L=L_{CE}+\alpha L_{GFD}+\beta L_{REG}
\end{equation}
where $\alpha$ and $\beta$ are hyper-parameters.

\section{Experiments}
\subsection{Experimental Settings}
\subsubsection{Datasets} 
We employed four graph datasets in our experiments: Income, Recidivism, Pokec-z \& Pokec-n. The income dataset, derived from the Adult dataset \cite{38}, identifies 'race' as the sensitive attribute with the task of predicting whether an individual's annual income exceeds \$50,000. The Recidivism dataset, sourced from Jordan et al. \cite{39}, also designates 'race' as the sensitive attribute, with the task of classifying whether a defendant is granted bail. The Pokec-z and Pokec-n datasets, collected from the popular Slovakian social network Pokec \cite{40}, list 'locating region' as the sensitive attribute, which refers to the geographical area where users are located, and the associated task is to classify the users' fields of employment. These datasets have been extensively utilized in previous studies on graph fairness learning, covering a diverse range of domains such as finance, criminal justice, and social network. The dataset statistics are shown in Table \ref{tab:results}.

\subsubsection{GNN Backbones}

In our experiments, we utilized two prominent GNNs as the encoding backbone: GCN \cite{7} and GIN \cite{30}. These two models are widely recognized and employed within the research community for their robust performance on various graph-related tasks.

\subsubsection{Baselines}

We compared our approach with four state-of-the-art methods for addressing fairness issues: FairGNN \cite{10}, NIFTY \cite{11}, EDITS \cite{12}, and BIND \cite{13}. FairGNN proposes a method that deploys an adversary to filter out sensitive attribute information from node embeddings to ensure the GNN classifier makes predictions independent of sensitive attribute, thereby achieving model debiasing. NIFTY proposes a novel framework that simultaneously accounts for fairness and stability,  introducing an objective function designed to optimize the alignment between the predictions based on perturbed sensitive attribute and unperturbed counterparts. EDITS focuses on input graph data, aiming to pre-process the data to achieve fairer GNNs through feeding the model with less biased data. BIND introduces a novel strategy for quantifying the influence of each training node on model bias, debiasing GNNs by deleting harmful nodes.

\subsection{Effectiveness Analysis}

\begin{table*}[h!]
\caption{Comparison among baseline methods with GFD. ($\uparrow$) denotes the larger, the better; ($\downarrow$) denotes the opposite. Best ones are in bold.}
\centering
\renewcommand{\arraystretch}{1.3}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|l|ccc|ccc|ccc|ccc}
\toprule
\multirow{2}{*}{\textbf{Encoder}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c|}{\textbf{Recidivism}} & \multicolumn{3}{c|}{\textbf{Income}} & \multicolumn{3}{c|}{\textbf{Pokec-n}} & \multicolumn{3}{c}{\textbf{Pokec-z}} \\ 
\cmidrule{3-14}
 &  & ($\uparrow$) ACC & ($\downarrow$) $\Delta_{SP}$ & ($\downarrow$) $\Delta_{EO}$ & ($\uparrow$) ACC & ($\downarrow$) $\Delta_{SP}$ & ($\downarrow$) $\Delta_{EO}$ & ($\uparrow$) ACC & ($\downarrow$) $\Delta_{SP}$ & ($\downarrow$) $\Delta_{EO}$ & ($\uparrow$) ACC & ($\downarrow$) $\Delta_{SP}$ & ($\downarrow$) $\Delta_{EO}$ \\
\midrule
\multirow{6}{*}{\textbf{GCN}} & Vanilla & \textbf{88.9±0.5} & 7.55±0.2 & 5.26±0.7 & 74.7±1.4 & 25.9±1.9 & 32.3±0.8 & 64.7±0.4 & 3.77±0.9 & 2.96±1.1 & 62.8±1.0 & 3.95±1.0 & 2.76±0.9 \\
 & FairGNN & 85.5±1.7 & 7.31±0.1 & 5.17±0.1 & 69.1±0.6 & 23.9±2.7 & 25.6±2.8 & 63.3±0.5 & 3.29±2.9 & 2.46±2.6 & \textbf{63.3±1.5} & 2.07±1.9 & 1.67±1.4 \\
 & NIFTY & 84.3±2.9 & 5.78±1.3 & 4.72±1.0 & 70.8±0.9 & 24.4±1.6 & 26.9±3.7 & 63.2±0.2 & 2.34±1.0 & 2.79±1.3 & 62.7±0.3 & 6.50±2.1 & 7.58±1.7 \\
 & EDITS & 87.3±0.1 & 6.64±0.3 & 4.51±0.9 & 68.3±0.8 & 24.0±1.9 & 25.9±1.7 & 64.6±0.9 & 2.52±0.8 & 2.61±1.6 & 62.8±0.9 & 2.75±1.8 & 2.24±1.5 \\
 & BIND & 88.7±0.0 & 7.40±0.0 & 5.09±0.1 & \textbf{75.2±0.0} & 19.2±0.6 & 26.4±0.4 & 63.5±0.4 & 6.75±2.3 & 5.41±3.4 & 60.6±0.8 & 5.85±2.0 & 1.15±0.7 \\
 & GFD & 88.0±0.9 & \textbf{5.31±0.3} & \textbf{3.82±0.9} & 74.2±1.2 & \textbf{15.7±2.0} & \textbf{19.3±2.1} & \textbf{64.8±0.5} & \textbf{1.96±1.5} & \textbf{1.92±1.7} & 62.8±1.2 & \textbf{1.46±1.5} & \textbf{1.07±1.2} \\
\midrule
\multirow{6}{*}{\textbf{GIN}} & Vanilla & 83.5±0.1 & 7.05±0.6 & 6.44±0.7 & 72.1±0.4 & 27.7±1.2 & 30.6±0.7 & \textbf{64.1±0.8} & 3.94±1.1 & 2.58±1.2 & 62.8±1.3 & 3.67±1.1 & 3.12±0.5 \\
 & FairGNN & 77.9±2.2 & 6.63±1.4 & 6.23±1.4 & 69.8±1.6 & 23.6±4.5 & 26.2±4.8 & 62.1±1.2 & 3.82±2.4 & 3.62±2.7 & \textbf{63.4±1.5} & 3.61±2.9 & 3.29±3.5 \\
 & NIFTY & 74.5±0.9 & 5.27±1.1 & 3.46±1.5 & 70.3±0.9 & 26.5±2.3 & 27.9±2.7 & 61.7±0.5 & 3.84±1.0 & 3.24±1.6 & 62.5±1.3 & 2.57±1.2 & 3.01±1.9 \\
 & EDITS & 83.6±0.1 & 5.02±0.3 & 2.89±0.1 & 68.8±0.8 & 25.8±1.8 & 27.2±1.7 & 59.8±0.9 & 2.75±0.8 & 2.24±1.5 & 61.9±0.3 & 3.33±0.8 & 3.65±1.6 \\
 & BIND & \textbf{83.7±0.1} & 6.94±0.3 & 5.66±0.2 & 71.5±0.3 & 21.2±0.9 & 25.7±1.0 & 63.6±0.1 & 6.68±1.9 & 5.32±2.2 & 62.7±0.1 & 3.76±0.9 & 1.99±1.5 \\
 & GFD & 83.1±0.1 & \textbf{4.79±0.5} & \textbf{2.06±1.0} & \textbf{72.1±0.1} & \textbf{18.8±2.0} & \textbf{20.2±1.7} & 63.2±0.7 & \textbf{1.45±1.1} & \textbf{2.12±1.6} & 62.5±1.0 & \textbf{1.64±1.8} & \textbf{1.16±2.3} \\
\bottomrule
\end{tabular}
}
\label{tab:results}
\end{table*}

We evaluated the performance of GFD against four baseline methods on a node classification task, using two different GNNs (GCN, GIN) as backbones. Table \ref{tab:results} presents the accuracy and fairness performance of our model compared to other debiasing approaches. From these results, we observed the following: (1) GFD demonstrates excellent fairness performance, which proves its effectiveness as a debiasing method. (2) GFD improves fairness performance without significantly sacrificing model accuracy. Although it may not always achieve the highest prediction accuracy compared to other state-of-the-art methods, its performance consistently ranks near the top in most scenarios. This demonstrates that GFD achieves a competitive fairness-utility trade-off. (3) GFD achieved excellent performance across all four datasets, demonstrating the strong adaptability and potential scalability of our method.

To further visualize the debiasing effect of GFD, we employed t-SNE to project the node embeddings from the Recidivism dataset into a 2-D space, as shown in Figure \ref{t-SNE}. The left plot displays embeddings learned by GCN without any debiasing, while the right plot demonstrates the debiased embeddings obtained by applying the GFD method. Node colors indicate race categories. Without debiasing, embeddings are distinctly clustered by sensitive attribute. In contrast, with GFD, embeddings with different sensitive attribute are blended together, demonstrating that our method effectively achieves model debiasing.

\begin{figure}[t]
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{without.png}
  \subcaption{GCN without GFD}
  \label{without}
\end{minipage}%
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{with.png}
  \subcaption{GCN with GFD}
  \label{with}
\end{minipage}%
\caption{Visualization of embeddings learned on Recidivism. Node color represents the sensitive attributes of nodes.}
\label{t-SNE}
\end{figure}

\subsection{Ablation Studies}

To investigate how the loss terms individually affect predictions, the fairness loss and the graph regularization loss were removed separately, representing the results as GFD\textbackslash F and GFD\textbackslash G. Since Accuracy Loss acts on the pretrained model, its removal would lead to uncontrollable changes in the results; therefore, further discussion on it is omitted. The ablation results of GFD on the Pokec-n and Pokec-z datasets are shown in Figure \ref{Ablation}, with GCN as the backbone. We observed that: (1) Removing the fairness loss will expand the bias, which proves the effectiveness of the fairness loss in debiasing. (2) Removing the graph regularization loss affects accuracy while slightly improving fairness, which demonstrates the value of introducing graph regularization for accuracy performance. Overall, simultaneously considering these loss terms better achieves the fairness-utility trade-off.

\begin{figure}[t]
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{Ablation_ACC.png}
  \subcaption{ACC performance}
  \label{without}
\end{minipage}%
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{Ablation_SP.png}
  \subcaption{$\Delta_{SP}$ performance}
  \label{with}
\end{minipage}%
\caption{Ablation results of loss terms in GFD. GFD\textbackslash F denotes the removal of fairness loss, and GFD\textbackslash G denotes the removal of graph regularization loss.}
\label{Ablation}
\end{figure}

\subsection{Efficiency Analysis}

We compared the training time costs of GFD against baseline models on the Recidivism and Income datasets, as shown in Fig \ref{Time}. While GFD does not achieve the lowest time cost, it demonstrates competitive efficiency compared to other topology debiasing methods, such as EDITS. Overall, although GFD is not the most efficient among all baseline methods, its time complexity remains within an acceptable range.

\begin{figure}[t]
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{Figure_1.png}
  \subcaption{Recidivism}
  \label{Recidivism}
\end{minipage}%
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{Figure_2.png}
  \subcaption{Income}
  \label{Income}
\end{minipage}%
\caption{Comparison of training time costs for GFD and baselines on the Recidivism and Income datasets (in seconds).}
\label{Time}
\end{figure}

\subsection{Hyper-parameter Analysis}

We conducted hyperparameter analysis on two parameters, $\alpha$ and $\beta$. Due to similar observations across different datasets, we specifically presented the parameter sensitivity results for the Recidivism dataset. Using GCN as the backbone, we vary $\alpha$ and $\beta$ within the ranges of \{0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 10, 100\} and \{0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10\}, respectively. As shown in Figure \ref{Parameters}, the following observations can be made: (1) the overall performance of GFD remains consistent despite the wide range of variations in $\alpha$ and $\beta$. Specifically, GFD maintains stable utility and fairness when $\alpha$ varies from 0.001 to 10 and $\beta$ from 0.5 to 10. (2) The best ACC performance is observed at $\alpha=0.3$ and $\beta=3$, while optimal $\Delta_{SP}$ and $\Delta_{EO}$ are achieved at $\alpha=100$ with $\beta=3$ and $\beta=5$, respectively. (3) When $\alpha$ is greater than or equal to 10, although GFD continues to improve fairness, the utility performance decreases significantly. When $\beta$ is less than or equal to 0.5, both fairness and utility performance sharply decline. This emphasizes the importance of selecting appropriate values for $\alpha$ and $\beta$ to optimize the fairness-utility trade-off, and highlights the cruciality of graph regularization in preventing unpredictable model behavior. Overall, GFD demonstrates stability with respect to the broad variation in $\alpha$ and $\beta$.

\begin{figure}[t]
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{1.png}
  \subcaption{ACC performance}
  \label{ACC}
\end{minipage}%
\begin{minipage}{0.5\linewidth}
  \centering
  \includegraphics[width=\linewidth]{2.png}
  \subcaption{$\Delta_{SP}$ performance}
  \label{SP}
\end{minipage}%
\caption{Hyper-parameter analysis on Recidivism.}
\label{Parameters}
\end{figure}

\section{Conclusion}

In this paper, we revisit topological bias in GNNs from a fine-grained perspective, with a focus on the influence of each edge on fairness. To achieve this, we first introduce the concept of Topological Fairness, which differentiates neighborhood information into appropriate and inappropriate information, and considers the latter as a confounding factor that introduces bias. Subsequently, to effectively differentiate between these two types of information and intervene in inappropriate information, we propose a novel method called GFD. By evaluating the sign of the derivative, we can determine whether the information is appropriate or inappropriate. By minimizing this value globally, we can mitigate topological bias and identify the optimal fairness graph. Our experiments on four widely used datasets achieve an excellent fairness-utility trade-off, demonstrating the effectiveness and competitiveness of our method. Overall, the definition of Topological Fairness lays the foundation for future research, and applying causal models to examine topological bias has the potential to open up the scope of research in this area.
## Implementation Details
The implementation is based on PyTorch. Specifically, we replicate each experiment using three different random seeds: 1, 10, and 100. All GNNs are optimized with Adam optimizer. The training is conducted for 1,000 epochs with learning rate chosen from \{0.1, 0.01, 0.001\}. The GNNs feature a hidden dimension of 16 and employ dropout rates selected from \{0.2, 0.5, 0.8\}. Additionally, we experiment with iteration numbers for Hessian matrix inverse estimation ranging from \{10, 50, 100, 500, 1000, 5000\} to optimize performance. All experiments are conducted on a device equipped with two GeForce RTX 4080 Super GPUs.
## Requirements
- **Python**: == 3.8.18
- **torch**: == 1.8.0 + cu111
- **cuda**: == 11.1
- **torch-geometric**: == 1.7.0
- **torch-scatter**: == 2.0.6
- **torch-sparse**: == 0.6.9
- **torch-spline-conv**: == 1.2.1
- **dgl**: == 0.6.1
- **scikit-learn**: == 0.23.1
- **numpy**: == 1.22.0
- **scipy**: == 1.4.1
- **networkx**: == 2.4
