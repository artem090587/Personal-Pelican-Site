Title: Galvanize - Week 07 - Day 1
Date: 2015-07-13 10:20
Modified: 2015-07-13 10:30
Category: Galvanize
Tags: data-science, galvanize, graphs, networks
Slug: galvanize-data-science-07-01
Authors: Bryan Smith
Summary: Today we covered graphs

#Galvanize Immersive Data Science

##Week 7 - Day 1

I missed this day because was I was at the [Data Science Summit](http://conf.dato.com/).   The quiz was on SQL statements, per usual, but I have not done them.

The lesson covered this day were network graphs, which I have now completed on my own.  This topic is interested to me because it gives sense of connectedness.   Unfortunately, these problems are computational intensive, assuming they are not NP Hard.

##Movie Graphs

We are going to investigate movies treating actors as nodes and movies as edges that connect actors.   We then are going to use a breath first search algorithm and return the shortest connection between two actors.

First we must read in the imbb data.


    from collections import defaultdict
    
    
    def load_imdb_data(filename):
        '''
        filename: name of imdb edge data file
    
        Read in the data and create two dictionaries of adjacency lists, one for
        the actors and one for the movies.
        '''
        f = open(filename)
        actors = defaultdict(set)
        movies = defaultdict(set)
        for line in f:
            actor, movie = line.strip().split('\t')
            actors[actor].add(movie)
            movies[movie].add(actor)
        f.close()
        return actors, movies
    
    actors, movies = load_imdb_data("data/imdb_edges.tsv")
    print len(actors), "Actors"
    print len(movies), "Movies"

    81290 Actors
    16753 Movies


To perform the breath-first search we need to make a queue to read from.  We will continue to added actors to the queue until we find a path between the two actors.   


    from Queue import Queue
    import time
    
    def shortest_path(actors,movies, actor1, actor2):
        
        q=Queue()
        
        if actor1 not in actors and actor2 not in actors:
            return None
        
        q.put((actor1,(actor1,)))
        
        while not q.empty():
            actor, path = q.get()
            if actor == actor2:
                return path
            for movie in actors[actor]:
                for next_actor in movies[movie]:
                    q.put((next_actor,path+(movie,next_actor)))
        return None
    
    path = shortest_path(actors, movies, actors.keys()[3],"Kevin Bacon")
    print path

    ('David Kagen', 'Shackles', 'Mandy June Turpin', 'Powder Blue', 'Forest Whitaker', 'The Air I Breathe', 'Kevin Bacon')



    def print_path(path):
        print "Number of Actors Between " + path[0] + " and " + path[-1] + " : " + str(len(path)/2 - 1)
        for i in range(len(path)):
            if i%2 == 0:
                print path[i]
            else:
                print "\t"+path[i]
    print_path(path)

    Number of Actors Between David Kagen and Kevin Bacon : 2
    David Kagen
    	Shackles
    Mandy June Turpin
    	Powder Blue
    Forest Whitaker
    	The Air I Breathe
    Kevin Bacon


It is possible that there are multiple paths between David Kagen and Kevin Bacon.  We can alter the code slightly to get each path.


    def all_shortest_path(actors,movies, actor1, actor2):
        
        q=Queue()
        found_match = False
        len_match = 0
        paths = []
        
        if actor1 not in actors and actor2 not in actors:
            return None
        
        q.put((actor1,(actor1,)))
        
        while not q.empty():
            actor, path = q.get()
            if actor == actor2:
                
                if len(paths) == 0:
                    len_match = len(path)
                    found_match = True
                    paths.append(path)   
                else:
                    if len(path) == len_match:
                        paths.append(path)     
            if not found_match:
                for movie in actors[actor]:
                    for next_actor in movies[movie]:
                        q.put((next_actor,path+(movie,next_actor)))
        
        if found_match:
            return paths
    
        return None
    
    
    paths = all_shortest_path(actors, movies, actors.keys()[3],"Kevin Bacon")
    print "Number of Paths: ", len(paths)

    Number of Paths:  40


## Dijkstra's shortest path algorithm

There are also weighted graphs that can be used to considering the cost of traveling between two nodes.   This is solved by Dijkstra's algorithm.  This can be implemented with the use of Priority Queues in our previous function



    class Vertex(object):
        def __init__(self, name):
            self.name = name
            self.neighbors = {}
    
        def add_neighbor(self, neighbor, weight):
            self.neighbors[neighbor] = weight
    
    
    class Graph(object):
        def __init__(self):
            self.vertices = {}
    
        def add_node(self, name):
            if name not in self.vertices:
                self.vertices[name] = Vertex(name)
    
        def add_edge(self, a, b, weight):
            self.add_node(a)
            self.add_node(b)
            self.vertices[a].add_neighbor(b, weight)
            self.vertices[b].add_neighbor(a, weight)
    
        def get_neighbors(self, node):
            if node in self.vertices:
                return self.vertices[node].neighbors
            return []
    
    
    def create_graph():
        g = Graph()
        g.add_edge('sunset', 'richmond', 4)
        g.add_edge('presidio', 'richmond', 1)
        g.add_edge('pac heights', 'richmond', 8)
        g.add_edge('western addition', 'richmond', 7)
        g.add_edge('western addition', 'pac heights', 2)
        g.add_edge('western addition', 'downtown', 3)
        g.add_edge('western addition', 'haight', 4)
        g.add_edge('mission', 'haight', 1)
        g.add_edge('mission', 'soma', 5)
        g.add_edge('downtown', 'soma', 5)
        g.add_edge('downtown', 'nob hill', 2)
        g.add_edge('marina', 'pac heights', 2)
        g.add_edge('marina', 'presidio', 4)
        g.add_edge('marina', 'russian hill', 3)
        g.add_edge('nob hill', 'russian hill', 1)
        g.add_edge('north beach', 'russian hill', 1)
        return g
    
    graph = create_graph()


    from Queue import PriorityQueue
    import time
    
    def weighted_shortest_path(start,end, graph):
        
        q=PriorityQueue()
        
        if start not in graph.vertices and end not in graph.vertices:
            return None
        
        q.put((start,(start,)),1)
        
        
        while not q.empty():
            node, path = q.get()
            if node == end:
                return path
            for next_node, next_weight in graph.get_neighbors(node).iteritems():
                q.put((next_node,path+(next_node,)),next_weight)
        return None
    
    weighted_shortest_path('presidio','pac heights',graph)




    ('presidio', 'marina', 'pac heights')



## Graphs & networkX

We are going to use the networkx package and redo the actor,movie analysis in networkx.


    import networkx as nx
    G = nx.read_edgelist('data/imdb_edges.tsv', delimiter='\t')

The package as a shortest path algorithm that is better implemented that what we wrote.  


    nx.shortest_path(G,'David Kagen','Kevin Bacon')




    ['David Kagen',
     u'Shackles',
     u'Mandy June Turpin',
     u'Powder Blue',
     u'Forest Whitaker',
     u'The Air I Breathe',
     'Kevin Bacon']



It also produces the same result as we got before, but the time is significantly faster.  

## Measures of Centrality and Connected Components

There are a few different ways to measure the most important nodes.  We are going to use a diffently formatter version of the data set and explore which actors are the most important.


    A = nx.read_edgelist('data/actor_edges.tsv', delimiter='\t')

One method of important is the degree of centrality in the network, and networkx has a method of calculating this.


    from collections import Counter
    Counter(nx.degree_centrality(A)).most_common(10)




    [(u'Danny Trejo', 0.013296281527265476),
     (u'Richard Riehle', 0.010330475525869803),
     (u'Keith David', 0.010131093609809589),
     (u'Tom Arnold', 0.009956634433256903),
     (u'Eric Roberts', 0.00968248429867411),
     (u'Michael Madsen', 0.00930864320606121),
     (u'David Koechner', 0.008286810886252617),
     (u'Christopher McDonald', 0.00814973581896122),
     (u'James Franco', 0.008025122121423587),
     (u'Paul Rudd', 0.007975276642408533)]



Thse are the most important actors according to this metric.  But we can see that all the actors are not networked together.  There are subgraphs in this network.


    nx.is_connected(A)




    False




    components = [len(component) for component in nx.connected_components(A)]
    print len(components)

    1390


In fact there are 1390 subgraphs in this dataset.  There is one large graph of 70k+ nodes. Lets look at the scale of these graphs


    import matplotlib.pyplot as plt
    import numpy as np
    %matplotlib inline
    plt.hist(components[1:],bins=15)
    plt.show()



![png](http://www.bryantravissmith.com/img/GW07D1/output_23_0.png)


Most of these results are a small group of actors, probably in a single movie.

There are other measure of centrality, but they are computationally intensive.  In order to explore them in networkx we were advise to use a smaller dataset.  


    SA = nx.read_edgelist('data/small_actor_edges.tsv', delimiter='\t')


    from collections import Counter
    Counter(nx.degree_centrality(SA)).most_common(10)




    [(u'David Koechner', 0.2556179775280899),
     (u'Justin Long', 0.25),
     (u'Danny Trejo', 0.2247191011235955),
     (u'Paul Rudd', 0.21629213483146068),
     (u'Jason Bateman', 0.20786516853932585),
     (u'Will Ferrell', 0.20786516853932585),
     (u'Stanley Tucci', 0.2050561797752809),
     (u'Samuel L. Jackson', 0.20224719101123595),
     (u'Elizabeth Banks', 0.199438202247191),
     (u'Woody Harrelson', 0.19662921348314605)]




    Counter(nx.betweenness_centrality(SA)).most_common(10)




    [(u'Danny Trejo', 0.017405842219758747),
     (u'Justin Long', 0.012435514894496458),
     (u'David Koechner', 0.011794132340103443),
     (u'John Goodman', 0.01165458626013287),
     (u'Samuel L. Jackson', 0.010876961690659508),
     (u'Robert De Niro', 0.00999681773729788),
     (u'Keith David', 0.009642697665367045),
     (u'Stanley Tucci', 0.009086485838244202),
     (u'Vinnie Jones', 0.009057085115156209),
     (u'Kris Kristofferson', 0.008697516162559957)]




    Counter(nx.eigenvector_centrality(SA)).most_common(10)




    [(u'David Koechner', 0.140462834180465),
     (u'Justin Long', 0.131152841463775),
     (u'Paul Rudd', 0.1284052374257351),
     (u'Will Ferrell', 0.12554389126285),
     (u'Seth Rogen', 0.11651549249985162),
     (u'Kristen Wiig', 0.11275491086152521),
     (u'Jason Bateman', 0.11125735405067895),
     (u'Ben Stiller', 0.10928205137571208),
     (u'Elizabeth Banks', 0.10659729338832805),
     (u'Jonah Hill', 0.10600648265284611)]



## Discovering Communities - Girvan Newman

We are going to implement the Girvan-Newman algorithm for discovering communities. Again, use the smaller graph so that are computations can be done quickly.

Here is the pseudocode for the Girvan-Newman algorithm:

```
function GirvanNewman:
    repeat:
        repeat until a new connected component is created:
            calculate the edge betweenness centralities for all the edges
            remove the edge with the highest betweenness
```

We are going to do this to find the 'optimal number' of communities in the data set.


    karateG = nx.karate_club_graph()
    nx.draw(karateG)


![png](http://www.bryantravissmith.com/img/GW07D1/output_30_0.png)


Below is the functions used to find the optimal number of communities in this network.


    import networkx as nx
    from collections import Counter
    
    
    def girvan_newman_step(G):
        '''
        INPUT: Graph G
        OUTPUT: None
    
        Run one step of the Girvan-Newman community detection algorithm.
        Afterwards, the graph will have one more connected component.
        '''
        num_connected_graphs = len([c for c in nx.connected_components(G)])
        while len([c for c in nx.connected_components(G)]) == num_connected_graphs:
            eb = nx.edge_betweenness(G)
            values = eb.values()
            edge_to_remove = eb.keys()[values.index(max(values))]
            G.remove_edge(edge_to_remove[0],edge_to_remove[1])
    
    
    def find_communities_n(G, n):
        '''
        INPUT: Graph G, int n
        OUTPUT: list of lists
    
        Run the Girvan-Newman algorithm on G for n steps. Return the resulting
        communities.
        '''
        G1 = G.copy()
        for i in xrange(n):
            girvan_newman_step(G1)
        return list(nx.connected_components(G1))
    
    
    def find_communities_modularity(G, max_iter=None):
        '''
        INPUT:
            G: networkx Graph
            max_iter: (optional) if given, maximum number of iterations
        OUTPUT: list of lists of strings (node names)
    
        Run the Girvan-Newman algorithm on G and find the communities with the
        maximum modularity.
        '''
        degrees = G.degree()
        num_edges = G.number_of_edges()
        G1 = G.copy()
        best_modularity = -1.0
        best_comps = nx.connected_components(G1)
        i = 0
        while G1.number_of_edges() > 0:
            subgraphs = nx.connected_component_subgraphs(G1)
            modularity = get_modularity(subgraphs, degrees, num_edges)
            if modularity > best_modularity:
                best_modularity = modularity
                best_comps = list(nx.connected_component_subgraphs(G1))
            girvan_newman_step(G1)
            i += 1
            if max_iter and i >= max_iter:
                break
        return best_comps
    
    
    def get_modularity(subgraphs, degrees, num_edges):
        '''
        INPUT:
            subgraphs: graph broken in subgraphs
            degrees: dictionary of degree values of original graph
            num_edges: float, number of edges in original graph
        OUTPUT: Float (modularity value, between -0.5 and 1)
    
        Return the value of the modularity for the graph G.
        '''
        q = 0.
        for g in subgraphs:
            nodes = g.nodes()
            edges = g.edges()
            n_nodes = len(nodes)
            n_edges = len(edges)
            sub_q = np.zeros((n_nodes,n_nodes)).astype(float)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        if (nodes[i],nodes[j]) in edges:
                            sub_q[i,j] += 1.
                        if (nodes[j],nodes[i]) in edges:
                            sub_q[i,j] += 1.
                        sub_q[i,j] -= float(degrees[nodes[i]]*degrees[nodes[j]])/(2*num_edges)
            sub_q = sub_q/(2*num_edges)
            q += sub_q.sum()
        return q


    communities = find_communities_modularity(karateG)
    communities




    [<networkx.classes.graph.Graph at 0x25cbeb0d0>,
     <networkx.classes.graph.Graph at 0x25cc46e90>,
     <networkx.classes.graph.Graph at 0x25cc46d90>,
     <networkx.classes.graph.Graph at 0x25cc46e10>,
     <networkx.classes.graph.Graph at 0x25cc46c10>]



Girvan Newman algorithm finds there are 5 communities.


    plt.figure(figsize=(14,8))
    for i,x in enumerate(communities):
        plt.subplot(2,3,i+1)
        nx.draw(communities[i])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW07D1/output_35_0.png)

