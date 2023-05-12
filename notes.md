Parker Dunn (pgdunn@bu.edu)  
April 2023

# Introductory Information

## Notes from Online

### Motivation


### Basics

GeeksForGeeks (https://www.geeksforgeeks.org/introduction-and-approximate-solution-for-vertex-cover-problem/)
* "A vertex cover of an undirected graph is a subset of its vertices s.t. for every edge (u, v) of the graph, either 
  'u' or 'v' is in the vertex cover."
* GOAL: Given an undirected graph, the vertex cover problem is to find a minimum size vertex cover.
* NP Complete problem - no polynomial-time solution for this unless P = NP
  * 

## Notes from Lecture

### Lecture 18 - 30 March 2023

#### Maximum matching & vertex cover

* Matching - subset of edges s.t. no edges share an endpoint (vertex)
  * **Maximum** matching - matching w/ maximal cardinality
  * **Maximal** matching - matching that cannot be extended w/o creating something that is no longer a matching
  * (# of edges in maximal matching) >= 1/2 * (# of edges in maximum matching)
* Vertex cover - subset of vertices s.t. each edge has at least one endpoint in the set
  * **Minimum** vertex cover - minimum cardinality vertex cover
* Maximal matching vs. minimum vertex cover
  * (# of edges in maximal matching) **<=** (# vertices in vertex cover) **<=** 2 * (# of edges in maximal matching)
  * Lower bound: "(# of edges in maximal matching) <= (# vertices in vertex cover)"
    * _EXTREME CASE EXAMPLE_: a STAR graph - only one edge is valid for maximum matching
  * Upper bound: "(# vertices in vertex cover) <= 2 * (# of edges in maximal matching)"
    * _EXTREME CASE EXAMPLE_: a FULLY-CONNECTED graph - all vertices can be in the vertex cover and only half of the 
      edges can be

#### Minimum vertex cover problem
* **GOAL**: Estimate the minimal vertex cover size
* Situation
  * C = small vertex cover
  * Oracle
    * IN: can ask if a vertex is in the vertex cover
    * OUT: Y/N
  * Algo
    * Asks Oracle if vertices are in 