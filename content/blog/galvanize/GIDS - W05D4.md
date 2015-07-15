Title: Galvanize - Week 05 - Day 4
Date: 2015-07-03 10:20
Modified: 2015-07-03 10:30
Category: Galvanize
Tags: data-science, galvanize, nlp, clustering, kmeans clustering, hierachrical cluster,
Slug: galvanize-data-science-05-04
Authors: Bryan Smith
Summary: Today we covered clustering

#Galvanize Immersive Data Science

##Week 5 - Day 4

Today we covered clutering.  Our morning quiz was a fun one because it involved A/B testing for 3 different pages.  We are told that the landing page was changed, and we have the number of useres that registered based on the page, and of those registered, we have the number who later purchased.   

The business goal is increasing sales, not registrations.   The page the converts the most people to buy is the ultimate goal of this test.  The data we were given is shown below.


|                | Visitors  | Registrations | Purchases |
|----------------|-----------|---------------|-----------|
| Landing Page 1 | 998,832   | 331,912       | 18,255    |
| Landing Page 2 | 1,012,285 | 349,643       | 18,531    |
| Landing Page 3 | 995,750   | 320,432       | 18,585    |

What is really cool is that we can perform a series of two sample hypothesis test and determine which page leads to the highest proportion of purchases.   The other method is that we can visualize the beta distribution and see which page leads to the most favoriable pdf.  


    import numpy as np
    import pandas as pd
    import scipy.stats as sp
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    x = np.linspace(0.0175,0.0195,10001)
    def plot_beta(x,success,total,label):
        y = sp.beta.pdf(x,success+1,total-success+1)
        plt.plot(x,y,label=label)
    plot_beta(x,18255,998832,"Page 1")
    plot_beta(x,18531,1012285,"Page 2")
    plot_beta(x,18585,995750,"Page 3")
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_1_0.png)


We can visually see that page 3 is better than page 1 and page 2, but this level of visualization does not capture how much better page 3 is than page 2 and page 1.   To do that we need to pick randomly from the distributions, and see how often page 3 is better than page 1 and page 2.


    page3betterthan2 = 0.
    page3betterthan1 = 0.
    sim_size = 100000
    for i in range(sim_size):
        if sp.beta.rvs(18585,995750) > sp.beta.rvs(18531,1012285):
            page3betterthan2 += 1.
        if sp.beta.rvs(18585,995750) > sp.beta.rvs(18255,998832):
            page3betterthan1 += 1.
            
    print page3betterthan1/sim_size, page3betterthan2/sim_size

    0.97701 0.96717


In our simulation, page 3 is statisically better than page 1 or page 2 are converting people who purchase products at the 5% level.   Our company should go with page 3.

If registrations was the metric, however, we might want to go with page two, as shown in the following plot.  


    x = np.linspace(0.31,0.35,1001)
    plot_beta(x,331912,998832,"Page 1")
    plot_beta(x,349643,1012285,"Page 2")
    plot_beta(x,320432,995750,"Page 3")
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_5_0.png)


## K-means

Our task for today is to create a k-means algorithm that has a random initialization of k centroids, then we recluster the data tot he closest centroids, move the centroid to the mean of the cluster, and repeat until there is convergence.

The random initalization will lead to different results from trial to trial.  To help mitigate this we will run the algorithm a fixed and setable number of times, and return the cluster with the lowest mean squared error.

We are going to try this data on the iris data set from sklearn.



    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    import numpy as np
    %matplotlib inline
    data = load_iris().data
    y = load_iris().target

I am calling my class kNice, cause i'm in a silly mood today!


    class kNice:
        
        def __init__(self,k,max_iter=100,n_runs=10):
            self.k = k
            self.max_iter = max_iter
            self.n_runs = n_runs
            self.best_clusters = None
            self.best_centroids = None
            self.best_sse = 1e17
            
        
        def fit(self,X):
            self.sse = 0
            self.X = X
            
            for i in range(self.n_runs):
                self.clusters = np.random.randint(0,self.k,size=X.shape[0])
                for i in range(self.k):
                    self.clusters[i] = i
                self.centroids = np.zeros((self.k,X.shape[1]))
                self._calculate_centroids()
                local_centroids = np.zeros((self.k,X.shape[1]))
                iters = 0
                
                while np.all(local_centroids!=self.centroids) and (iters <= self.max_iter):
                    local_centroids = self.centroids.copy()
                    self._update_centroids()
                    iters += 1
                sse = self._sse()
                if sse < self.best_sse:
                    self.best_sse = sse
                    self.best_clusters = self.clusters
                    self.best_centroids = self.centroids
            
            self.centroids = self.best_centroids
            self.clusters = self.best_clusters
            self.sse = self.best_sse
                
        def _sse(self):
            sse = 0
            for i in range(self.k):
                if np.any(np.isnan(self.centroids[i])):
                    sse = 1e16
                    break
                else:
                    sse += np.sum(np.power(self.X[self.clusters==i]-self.centroids[i] ,2))
            return sse
        
            
            
        def _calculate_centroids(self):
            for i in range(self.k):
                self.centroids[i] = np.mean(self.X[self.clusters==i,:],axis=0)
                if np.any(np.isnan(self.centroids[i])):
                    self.centroids[i] = self.X[np.random.randint(0,self.X.shape[0])]
                    
                
        def _update_centroids(self):
            if self.k > 1:
                results = np.linalg.norm(self.X-self.centroids[0],axis=1)
                for i in range(1,self.k):
                    results = np.vstack((results,np.linalg.norm(self.X-self.centroids[i],axis=1)))
                self.clusters = np.argmin(results,axis=0)
                self._calculate_centroids()
        
    clus = kNice(k=3,n_runs=10)
    
    clus.fit(data)
    
    for target in np.unique(y):
        mask = y==target
        plt.plot(data[mask,0],data[mask,1],marker='o',lw=0)
    for x in clus.centroids:
        plt.plot(x[0],x[1],'ys',markersize=25)
    print "SSE: ", clus.sse

    SSE:  78.9408414261



![png](http://www.bryantravissmith.com/img/GW05D4/output_9_1.png)


We can see in this case that 3 clusters seems to match the different types of iris flowers.   If we did not already know this, we can use the elbow method of the sum of square errors (SSE), and pick the number of clusters where the elbow is.


    for i in range(1,20):
        clus = kNice(i)
        clus.fit(data)
        plt.plot(i,clus.sse,'ro')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_11_0.png)


We see that this happens to be n=3, which is also the number of clusters that are in the data set.   There is also the silhouette score from sklearn that finds the mean intercluster distance.  The best value is 1, and the worst values is -1.  


    from sklearn.metrics import silhouette_score
    for i in range(2,20):
        clus = kNice(i)
        clus.fit(data)
        plt.plot(i,silhouette_score(data, clus.clusters, metric='euclidean'),'ro')
    plt.show()



![png](http://www.bryantravissmith.com/img/GW05D4/output_13_0.png)


We see that in this case, the best cluster size is 2, followed by 3 and so forth.   With the overlap of the two of the iris data, it makes sense that two clusters would be better by this metric.

##Random 3D Plot.

Its the day before we leave for the 4th of July.   I feel like plotting 3D!


    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    
    from sklearn.cluster import KMeans
    from sklearn import datasets
    
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    fig = plt.figure(1, figsize=(14, 13))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=12, azim=134)
    
    plt.cla()
    nice = kNice(3)
    nice.fit(X)
    labels = nice.clusters
    
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    plt.show()
     


![png](http://www.bryantravissmith.com/img/GW05D4/output_15_0.png)


##NY Times - Clustering

In the afternoon spirt, we are trying to identify topics in the New York Times by clustering.   We have 1400+ articles saved in a pickle file.   We will read them in, TFIDF them, and cluster.  


    df = pd.read_pickle("data/articles.pkl")
    df.shape




    (1405, 15)




    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import re
    from nltk import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.snowball import SnowballStemmer
    
    def tokenize(doc):
        '''
        INPUT: string
        OUTPUT: list of strings
    
        Tokenize and stem/lemmatize the document.
        '''
        sw = set(stopwords.words('english'))
        snowball = SnowballStemmer('english')
        reg = RegexpTokenizer(r'\w+',flags=re.UNICODE)
        doc_tokens = []
        for t in reg.tokenize(doc):
            if t not in sw:
                s = snowball.stem(t)
                doc_tokens.append(s)    
        return doc_tokens
    
    
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize,stop_words='english',strip_accents='unicode',max_features=10000)
    tfidf = vectorizer.fit_transform(df.content.values)


    num_clust = 20
    clus = KMeans(num_clust)
    clus.fit(tfidf)
    for i in range(num_clust):
        print ""
        print "Cluster: " + str(i)
        temp = np.zeros(clus.cluster_centers_[i].shape)
        temp[np.argsort(clus.cluster_centers_[i])[-15:]] = 1
        temp = temp/np.linalg.norm(temp)
        print "top 10 words"
        print vectorizer.inverse_transform(temp)[0]
        print "top 5 headlines"
        print df.headline[clus.labels_==i][:5]
        

    
    Cluster: 0
    top 10 words
    [u'care' u'cruz' u'debt' u'democrat' u'govern' u'health' u'hous' u'law'
     u'mr' u'obama' u'parti' u'republican' u'senat' u'shutdown' u'vote']
    top 5 headlines
    1                      New Immigration Bill Put Forward
    17                                  Congress Breaks Bad
    18                            Excuses, Excuses, Excuses
    48    Obama Sets Conditions for Talks: Pass Funding ...
    78                            Our Democracy Is at Stake
    Name: headline, dtype: object
    
    Cluster: 1
    top 10 words
    [u'buffalo' u'defens' u'game' u'holm' u'jet' u'lankster' u'quarterback'
     u'ryan' u'said' u'season' u'smith' u'titan' u'turnov' u'week' u'yard']
    top 5 headlines
    55                  Titans Quarterback Out a ‘Few Weeks’
    129                     Time for Smith to Learn, Quickly
    231                                        Jets Close-Up
    253    Tennessee Turns Smith’s Giveaways Into Touchdowns
    329                           Jets (2-1) at Titans (2-1)
    Name: headline, dtype: object
    
    Cluster: 2
    top 10 words
    [u'attack' u'chemic' u'govern' u'kill' u'mr' u'offici' u'peopl' u'polic'
     u'rebel' u'said' u'secur' u'syria' u'syrian' u'unit' u'weapon']
    top 5 headlines
    22                   Libya: Mob Attacks Russian Embassy
    38    Man Charged With False Report After Airport Cl...
    42     Missed Opportunity in Syria Haunts U.N. Official
    62                      Airport in Florida Is Evacuated
    82    Libya: 27 Tortured to Death in Jails Run by Mi...
    Name: headline, dtype: object
    
    Cluster: 3
    top 10 words
    [u'coach' u'game' u'hit' u'inning' u'l' u'leagu' u'n' u'play' u'player'
     u'run' u'said' u'score' u'season' u'team' u'win']
    top 5 headlines
    6               Bayern Munich Dominates Manchester City
    8                      Brodeur’s Starting Streak to End
    21    Finally Secure in the Desert, the Coyotes Devo...
    26    Braves’ Free Swingers Face Dodgers’ Staff of Aces
    35       A’s Follow Crisp’s Lead Back Into the Playoffs
    Name: headline, dtype: object
    
    Cluster: 4
    top 10 words
    [u'coalit' u'dawn' u'democrat' u'elect' u'europ' u'german' u'germani'
     u'golden' u'govern' u'merkel' u'mr' u'parliament' u'parti' u'polit'
     u'said']
    top 5 headlines
    34     Risk-Averse Gandhi’s Move Rattles Indian Election
    44      Case Against Greek Far-Right Party Draws Critics
    49             Mutiny Halts Italian Gambit by Berlusconi
    95     Internal Dissent Imperils Berlusconi’s Long Re...
    154                    Norway: New Coalition Tilts Right
    Name: headline, dtype: object
    
    Cluster: 5
    top 10 words
    [u'0' u'1' u'2' u'3' u'coughlin' u'defens' u'game' u'giant' u'offens'
     u'play' u'quarterback' u'season' u'team' u'touchdown' u'yard']
    top 5 headlines
    0      Week 5 Probabilities: Why Offense Is More Impo...
    73     Woes Continue for Giants as Snee Considers Sur...
    131         Nowhere to Look but Ahead for Winless Giants
    134              Saints Hand Dolphins Their First Defeat
    225    Week 4 Quick Hits: Corners Cover for Patriots’...
    Name: headline, dtype: object
    
    Cluster: 6
    top 10 words
    [u'citi' u'court' u'like' u'mr' u'ms' u'new' u'peopl' u'race' u'said'
     u'school' u'sept' u'state' u'time' u'world' u'year']
    top 5 headlines
    2    Arizona: Judge Orders Monitor to Oversee Maric...
    3    Texas: State Bought Execution Drugs From a Com...
    4                        Nadal on Track for No. 1 Spot
    7        American Leads in World Gymnastics All-Around
    9                           Vonn Is Close to Returning
    Name: headline, dtype: object
    
    Cluster: 7
    top 10 words
    [u'basebal' u'cano' u'game' u'girardi' u'inning' u'jeter' u'mariano'
     u'pettitt' u'pitch' u'rivera' u'rodriguez' u'said' u'season' u'stadium'
     u'yanke']
    top 5 headlines
    87                        YES Viewers Say Their Goodbyes
    243    For Girardi, Yanks’ Goodbyes Won’t Get Any Easier
    263    Memorable End for Two; Forgettable Year for Yanks
    326      Rivera Says There Will Be No Encore Performance
    331    With a Win, the Ending to Pettitte’s Illustrio...
    Name: headline, dtype: object
    
    Cluster: 8
    top 10 words
    [u'212' u'art' u'artist' u'citi' u'design' u'exhibit' u'galleri' u'like'
     u'm' u'mr' u'museum' u'paint' u'said' u'street' u'work']
    top 5 headlines
    13     On Fashion Runway, South Sudan Takes Steps Tow...
    31                 Surrounding Art With the Sounds of 60
    41      Midnight at the Museum, Breakfast at the Vatican
    130             A Monument to the West That Many Pass By
    175           Storytelling Coming to Life at the Library
    Name: headline, dtype: object
    
    Cluster: 9
    top 10 words
    [u'america' u'boat' u'club' u'cup' u'francisco' u'golf' u'oracl' u'race'
     u'regatta' u'said' u'san' u'spithil' u'team' u'yacht' u'zealand']
    top 5 headlines
    5                  Judge Halts Work on World Cup Stadium
    10           Whitney Winner Out of Breeders’ Cup Classic
    125    Australia Primed for First Cup Challenge Since...
    311        Presidents Cup Is Custom-Made for Tiger Woods
    312    Nick Price Has a Mission to Motivate at Presid...
    Name: headline, dtype: object
    
    Cluster: 10
    top 10 words
    [u'carbon' u'china' u'coal' u'emiss' u'gas' u'japan' u'korea' u'korean'
     u'mr' u'north' u'nuclear' u'oil' u'plant' u'said' u'south']
    top 5 headlines
    12              Fuel From Landfill Methane Goes on Sale
    23               Evidence North Korea Restarted Reactor
    28    Another Shutdown Victim: U.S. Efforts to Offse...
    45    Back in Asia, Hagel Pursues Shift to Counter C...
    50    U.S. and South Korea Set Defense Strategy for ...
    Name: headline, dtype: object
    
    Cluster: 11
    top 10 words
    [u'iran' u'iranian' u'israel' u'mr' u'nation' u'negoti' u'netanyahu'
     u'nuclear' u'obama' u'presid' u'rouhani' u'said' u'sanction' u'state'
     u'unit']
    top 5 headlines
    20                Iran’s President Responds to Netanyahu
    149               Iran Staggers as Sanctions Hit Economy
    261    Amid Nuclear Issue, Israel Said to Arrest Iran...
    310       Dueling Narratives in Iran Over U.S. Relations
    323    Israel and Others in Mideast View Overtures of...
    Name: headline, dtype: object
    
    Cluster: 12
    top 10 words
    [u'afford' u'care' u'employe' u'exchang' u'feder' u'govern' u'health'
     u'hous' u'insur' u'law' u'new' u'republican' u'said' u'state' u'worker']
    top 5 headlines
    15                                 New York: Two Cities?
    16                      A Blind Spot on Rearview Cameras
    37     As Demand Stays High, Officials Try to Address...
    98       Justices to Hear ‘Raging Bull’ Copyright Appeal
    105     Invitation to a Dialogue: Distrust of Government
    Name: headline, dtype: object
    
    Cluster: 13
    top 10 words
    [u'album' u'band' u'drake' u'dylan' u'festiv' u'guitar' u'like' u'lyric'
     u'mr' u'music' u'play' u'pop' u'record' u'song' u'sound']
    top 5 headlines
    89                           Movement Made for Listening
    174        At a Festival, Molly Danced but Didn’t Cut In
    177    Congressman Proposes New Rules for Music Royal...
    191    Traditional Folk Frolic, With Old-Time Fervor ...
    193                 The Blow and Haim Release New Albums
    Name: headline, dtype: object
    
    Cluster: 14
    top 10 words
    [u'compani' u'concert' u'levin' u'met' u'mr' u'ms' u'music' u'new' u'night'
     u'opera' u'orchestra' u'perform' u'said' u'season' u'work']
    top 5 headlines
    33       Praying to the Moon, While Lashing Out at Fate
    39       Minnesota: Orchestra Cancels New York Concerts
    52    Fact Slides Into Fiction as National Theater V...
    90    It’s No Longer on His Face, and It’s Got Ambit...
    93               A Plucky Opera’s Poignant Death Rattle
    Name: headline, dtype: object
    
    Cluster: 15
    top 10 words
    [u'abc' u'cbs' u'drama' u'emmi' u'fox' u'hbo' u'm' u'nbc' u'network' u'p'
     u'rate' u'season' u'seri' u'televis' u'viewer']
    top 5 headlines
    29                       Family Trees of Familiar Scions
    203    More Than 10 Million Watch Finale of ‘Breaking...
    226                                     What’s on Monday
    257                                     What’s On Sunday
    427                                   What’s On Saturday
    Name: headline, dtype: object
    
    Cluster: 16
    top 10 words
    [u'arctic' u'attack' u'greenpeac' u'kenya' u'kenyan' u'mall' u'milit'
     u'nairobi' u'offici' u'said' u'shabab' u'ship' u'somali' u'somalia'
     u'westgat']
    top 5 headlines
    63     During Siege at Kenyan Mall, Government Forces...
    207          Somali Militants Mixing Business and Terror
    394    Somali Community in U.S. Fears New Wave of Sti...
    442            Tale of ‘White Widow’ Fills British Press
    444    Narrow Escapes and Questions on Emergency Resp...
    Name: headline, dtype: object
    
    Cluster: 17
    top 10 words
    [u'0' u'bank' u'econom' u'economi' u'euro' u'govern' u'growth' u'index'
     u'investor' u'market' u'percent' u'price' u'rate' u'said' u'zone']
    top 5 headlines
    46     With Inflation Low, the European Central Bank ...
    72     Growth Forecast Is Trimmed for Developing Nati...
    106       Vatican Bank Publishes Its First Annual Report
    112    European Unemployment Steady, Hinting at Progr...
    118      Japan Sales Tax to Increase Next Year, Abe Says
    Name: headline, dtype: object
    
    Cluster: 18
    top 10 words
    [u'advertis' u'agenc' u'brand' u'c' u'chief' u'compani' u'e' u'execut'
     u'jpmorgan' u'market' u'mr' u'new' u'pay' u'said' u'vice']
    top 5 headlines
    30     Peter Schlessel to Lead Universal’s Focus Feat...
    81       Groceries Are Cleaning Up in Store-Brand Aisles
    86     Worried About Land Grabs, Group Presses 3 Corp...
    108         Merck Plans to Lay Off Another 8,500 Workers
    110    HarperCollins Joins Scribd in E-Book Subscript...
    Name: headline, dtype: object
    
    Cluster: 19
    top 10 words
    [u'artist' u'ballet' u'choreograph' u'danc' u'dancer' u'mr' u'ms' u'music'
     u'new' u'perform' u'piec' u'program' u'stage' u'theater' u'work']
    top 5 headlines
    279             A Swing and a Jaunt Around the Globe
    282    An Impish Faun, Swans, and a Lively Card Deck
    335                Another Title for Japanese Skater
    415    After 45 Years, Freedom Salted With Hindsight
    421                   Interpreting ‘Rite’ in 2 Parts
    Name: headline, dtype: object


This clustering method works pretty well for establishing topics, but we need to label them by hand.  Also the number of times we run it we get different results.

We also do not know the correct number of clusters.  Above we have 20 clusters that seem well defined.  Below we have 3 clusters that also seem well defined.   This is where we get into art and not science.


    num_clust = 3
    clus = KMeans(num_clust)
    clus.fit(tfidf)
    for i in range(num_clust):
        print ""
        print "Cluster: " + str(i)
        temp = np.zeros(clus.cluster_centers_[i].shape)
        temp[np.argsort(clus.cluster_centers_[i])[-15:]] = 1
        temp = temp/np.linalg.norm(temp)
        print "top 10 words"
        print vectorizer.inverse_transform(temp)[0]
        print "top 5 headlines"
        print df.headline[clus.labels_==i][:5]

    
    Cluster: 0
    top 10 words
    [u'chemic' u'iran' u'iranian' u'israel' u'mr' u'nation' u'nuclear' u'obama'
     u'presid' u'rouhani' u'said' u'syria' u'syrian' u'unit' u'weapon']
    top 5 headlines
    20                Iran’s President Responds to Netanyahu
    42      Missed Opportunity in Syria Haunts U.N. Official
    96     Citing Efforts to Prevent Attack on Syria, Gro...
    149               Iran Staggers as Sanctions Hit Economy
    209    Invoking Sept. 11, Syrian Accuses U.S. of Hypo...
    Name: headline, dtype: object
    
    Cluster: 1
    top 10 words
    [u'0' u'1' u'game' u'leagu' u'n' u'play' u'player' u'rivera' u'run' u'said'
     u'season' u'team' u'win' u'yanke' u'yard']
    top 5 headlines
    0     Week 5 Probabilities: Why Offense Is More Impo...
    6               Bayern Munich Dominates Manchester City
    8                      Brodeur’s Starting Streak to End
    10          Whitney Winner Out of Breeders’ Cup Classic
    21    Finally Secure in the Desert, the Coyotes Devo...
    Name: headline, dtype: object
    
    Cluster: 2
    top 10 words
    [u'compani' u'govern' u'hous' u'like' u'mr' u'ms' u'new' u'parti' u'peopl'
     u'percent' u'republican' u'said' u'state' u'work' u'year']
    top 5 headlines
    1                     New Immigration Bill Put Forward
    2    Arizona: Judge Orders Monitor to Oversee Maric...
    3    Texas: State Bought Execution Drugs From a Com...
    4                        Nadal on Track for No. 1 Spot
    5                Judge Halts Work on World Cup Stadium
    Name: headline, dtype: object


## Hierarchical Clustering

We have been working with kmean's clustering, that relies heavily on a distance metric.  This is a general problem in machine linearing.   We have seen the importance of picking the right metric for the right problem.   The number of clusters, however, has been a variable that we do not need to pick.  Hierarchical Clustering is a method of clustering that breaks the data into subgroups, then subgroups into smaller subgroups, once a metric is determined.   In there no apriori reason to pick a cluster size. 

We are going to do this on the Time's data.


    articles = []
    for section in df.section_name.unique()[:5]:
        print section,len(df[df.section_name==section].content.values)
        for i in range(20):
            articles.append(df[df.section_name==section].content.values[i])
    print len(articles)
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize,stop_words='english',strip_accents='unicode')
    tfidf = vectorizer.fit_transform(articles)
    tfidf

    Sports 340
    U.S. 190
    Business Day 209
    World 260
    Opinion 224
    100





    <100x5488 sparse matrix of type '<type 'numpy.float64'>'
    	with 15919 stored elements in Compressed Sparse Row format>




    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    
    distances = squareform(pdist(tfidf.todense()))
    linkage(distances)
    plt.figure(figsize=(14,20))
    ax = plt.gca()
    R = dendrogram(linkage(distances),p=50,truncate_mode='level',
                   leaf_label_func=lambda x: df[df.index==x].headline.values[0].encode('ascii','ignore') + 
                       " |||| " + df[df.index==x].section_name.values[0].encode('ascii','ignore'),
                   ax=ax,
                   orientation='left',
                   leaf_font_size=12,
                   no_plot=False)
    plt.xlim([1.4,2.])
    plt.show()



![png](http://www.bryantravissmith.com/img/GW05D4/output_24_0.png)



    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    
    distances = squareform(pdist(tfidf.todense()))
    linkage(distances)
    plt.figure(figsize=(14,20))
    ax = plt.gca()
    R = dendrogram(linkage(distances),p=500,truncate_mode='level',
                   leaf_label_func=lambda x: df[df.index==x].section_name.values,
                   ax=ax,
                   orientation='left',
                   leaf_font_size=8,
                   no_plot=False)
    plt.xlim([1.2,2])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_25_0.png)



    distances = squareform(pdist(tfidf.todense(),metric='cosine'))
    linkage(distances)
    plt.figure(figsize=(14,20))
    ax = plt.gca()
    R = dendrogram(linkage(distances),p=25,truncate_mode='level',
                   leaf_label_func=lambda x: df[df.index==x].section_name.values + " - " + df[df.index==x].headline.values,
                   ax=ax,
                   orientation='left',
                   leaf_font_size=20,
                   no_plot=False)
    
    plt.xlim([0.4,1.45])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_26_0.png)



    distances = squareform(pdist(tfidf.todense(),metric='braycurtis'))
    linkage(distances)
    plt.figure(figsize=(14,20))
    ax = plt.gca()
    R = dendrogram(linkage(distances),p=25,truncate_mode='level',
                   leaf_label_func=lambda x: df[df.index==x].section_name.values + " - " + df[df.index==x].headline.values,
                   ax=ax,
                   orientation='left',
                   leaf_font_size=20,
                   no_plot=False)
    
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_27_0.png)



    distances = squareform(pdist(tfidf.todense(),metric='hamming'))
    linkage(distances)
    plt.figure(figsize=(14,20))
    ax = plt.gca()
    R = dendrogram(linkage(distances),p=25,truncate_mode='level',
                   leaf_label_func=lambda x: df[df.index==x].section_name.values + " - " + df[df.index==x].headline.values,
                   ax=ax,
                   orientation='left',
                   leaf_font_size=20,
                   no_plot=False)
    
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_28_0.png)


We can see immediately that the metric changes the results of the clustering.  We can use the linkage to cluster articles together because it lists the index of each cluster, as well as the distance between clusteres.   It is a different, and computationally more expensive, way to do clustering.

We can also take the transpose of the tfidf matrix to do clustering by words.  To do this we will remake the tfidf matrix and limit the number of words.


    vectorizer = TfidfVectorizer(tokenizer=tokenize,stop_words='english',strip_accents='unicode',min_df=0.2,max_features=1000)
    tfidf = vectorizer.fit_transform(df.content.values)


    vocab = dict()
    for k,v in vectorizer.vocabulary_.iteritems():
        vocab[v]=k


    distances = squareform(pdist(tfidf.todense().T,metric='cosine'))
    plt.figure(figsize=(14,20))
    ax = plt.gca()
    R = dendrogram(linkage(distances),p=100,truncate_mode='level',
                   leaf_label_func=lambda x: vocab[int(x)] if x in vocab else x,
                   ax=ax,
                   orientation='left',
                   #leaf_rotation=90,
                   leaf_font_size=8,
                   no_plot=False)
    
    
    plt.xlim([0.6,1.2])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D4/output_32_0.png)


There are some interesting connections.   "State", "nation", and "president" are cluster together.   So is "game" and "season". We limited this to a reasonable number of words, but there are clear word clusters in the corpus.

Since the day is now over - and tomorrow is the 4th of July Holiday.  I'm off!


    
