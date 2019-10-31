# Deep Reinforcement Learning for Polynomial Combinatorial Optimization
ORIE 4741 Final Project
Brandon Kates(bjk224), Peter Haddad (ph387), Michael Lapolla (mel259)


<p><b>Problem Statement</b></p>
<p>Deep Neural Networks are ubiquitous for solving Natural Language Processing and Computer Vision tasks, and are revered as being able to solve any problem given the right model. There has been a clash between traditional optimization algorithms, which have rigorous mathematical proofs showing how they work, and deep learning ‘black box’ algorithms which work extremely well but which are much more difficult to understand. We want to evaluate the ability of deep learning models on more traditional optimization problems, like finding the minimum spanning tree of a graph, to see how they perform given some constraints.</p>

<p><b>Motivation and Background</b></p>
<p>We want to understand the limitations of DL (deep learning). We want to know how difficult it would be to solve very simple combinatorial problems with architectures intended for more complex problems. We want to continue on prior work that has been done in the area to see where we can push it. Graph Neural Networks (GNN’s) have been applied to the traveling salesman problem (TSP) and Dijkstra’s, but none that we found specifically for the minimum spanning tree algorithm.</p>

<p><b>Proposed Approach / Proposed Experiments</b></p>
<p>We propose to build a Graph Neural Network architecture that can take in a graph (defined as G=(V,E)) and solve the minimum spanning tree algorithm. We want to see how our architecture compares to Prim’s and Kruskal’s algorithm for solving minimum spanning trees, given a fixed amount of training time, a fixed number of examples, or other limited resources.</p>

<p>We first need to solve the problem of generating random graphs that we can then feed into our network. Once we are able to generate random graphs of any size and find the optimal minimum spanning tree solution, we can evaluate the performance of our model architecture on the various features above. We would like to observe how the model learns, and if it follows a similar structure to the combinatorial solutions, if it comes up with its own way of learning, or if it is not able to learn the solutions.</p>

<p>Once we are able to apply DL to the minimum spanning tree problem, we will move on to applying the same architecture to other simple combinatorial problems (like Dijkstra’s and Ford-Fulkerson).</p>


<p><b>Resources:</b></p>

[Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)
