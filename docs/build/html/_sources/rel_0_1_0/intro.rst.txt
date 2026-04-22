Introduction to Temporal GNN
==============================================

There are so many real-word systems that can be formulated as temporal interaction graphs, such as social network and citation network. In these systems, the nodes represent the entities and the edges represent the interactions between entities. The interactions are usually time-stamped, which means the edges are associated with time. Temporal interaction graphs are dynamic, which means the graph structure changes over time. For example, in a social network, the friendship between two people may be established or broken at different time. In a citation network, a paper may cite another paper at different time. 

To encapsulate the temporal information present in these graphs and learn dynamic representations, researchers have introduced temporal graph neural networks (GNNs). These networks are capable of modeling both structural and temporal dependencies within the graph. Numerous innovative frameworks have been proposed to date, achieving outstanding performance in specific tasks such as link prediction. Based on two different methods to represent temporal graphs, we can divide temporal GNNs into two categories: 

1. continuous-time temporal GNNs, which model the temporal graph as a sequence of interactions
2. discrete-time temporal GNNs, which model the temporal graph as a sequence of snapshots

However, as the temporal graph expands—potentially encompassing millions of nodes and billions of edges—it becomes increasingly challenging to scale temporal GNN training to accommodate these larger graphs. The reasons are twofold: first, sampling neighbors from a larger graph demands more time; second, chronological training also incurs a higher time cost. To address these challenges, we introduce StarryGL in this tutorial. StarryGL is a distributed temporal GNN framework designed to efficiently navigate the complexities of training larger temporal graphs.