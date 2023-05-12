# Notes for Implementing Vertex Cover Approximation from In-Class (Lecture 19 - 21)

## Model

This is the assumed capability of the system that I am using. In other words, these are methods that must be written 
by me and made available to the algorithm.

1. Ability to draw a uniformly random vertex in the graph
2. A method that returns the degree of any vertex

## Partitioning Vertices into Buckets

### How we partition...

The core concept of this approach is partitioning the vertices in the graph based on the degree of the vertex. The 
bucketing method is as follows:

$$B_i = \{v \in V: (1 + \frac{\epsilon}{10})^{i} \leq deg(v) < (1 + \frac{\epsilon}{10})^{i+1}\}$$

where `deg(v)` is the degree of vertex `v`.

This partitioning approach assumes `deg(v) > 0` for all vertices

*This results in* $O(\frac{1}{\epsilon}log(n))$ buckets

### How buckets are used...

Two groups of buckets:

1. Heavy buckets: $B_{i}$ is heavy if $|B_{i}| \geq T$
2. Light buckets: $B_{i}$ is light if $|B_{i}| < T$

We end up with two sets:

1. $H = \{v \in V : v \in \text{heavy } B_{i}\}$
2. $L = \{v \in V : v \in \text{lgith } B_{i}\}$

Based on these two sets, we can re-formulate what `deg(v)` represents (since we have two types of vertices now).

$$deg(v) = d_{H}(v) + d_{L}(v)$$

where...

* $d_{L}(v)$ is the number of edges to vertices in $H$
* $d_{H}(v)$ is the number of edges to vertices in $L$

### How buckets relate to number of edges and vertex cover

With our bucketing formulation, we reconsider how you can calculate the number of edges:

$$|E| = \frac{1}{2} \sum_{v \in V} deg(v) = \frac{1}{2} (\sum_{v \in H}d_{H}(v) + \sum_{v \in H}d_{L}(v) + \sum_{v \in L}d_{H}(v) + \sum_{v \in L}d_{L}(v))$$

... running low on time, so I am not going to finish re-writing the algorithm right now

## Algorithm Outline

INPUTS

* $\epsilon$ - a parameter that controls the size of the buckets
* 