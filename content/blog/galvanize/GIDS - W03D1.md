Title: Galvanize - Week 03 - Day 1
Date: 2015-06-15 10:20
Modified: 2015-06-15 10:30
Category: Galvanize
Tags: data-science, galvanize, EDA, Linear Algebra
Slug: galvanize-data-science-03-01
Authors: Bryan Smith
Summary: The eleventh day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered EDA and Linear Algebra

#Galvanize Immersive Data Science

##Week 3 - Day 1

Today we had the first quiz I did not finish.  I attempted to do algebra in latex, and made a sign mistake and had difficulty traking my mistake.   

The morning involved using numpy to solve some linear algrebra based problems.  The afternoon had to do with exploratory data analysis and linear regression.   

##Miniquiz

Probability Practice

Let's say we play a game where I keep flipping a coin until I get heads. If the first time I get heads is on the nth coin, then I pay you 2n-1 dollars. How much would you pay me to play this game for us to break even in the long term? Show your work.

P(n) = p^(n-1)*(1-p)

$$E(V) = \Sigma_{i=1}^{\infty} p^{n-1} \ (1-p) \ (2n-1) $$

$$E(V) = \Sigma_{i=1}^{\infty} (2n-1) (p^{n-1} - p^n) $$

$$E(V) = \Sigma{i=1}^{\infty} 2 \ n \ p^{n-1} - \Sigma{i=1}^{\infty} 2 \ n \ p^{n} - \Sigma{i=1}^{\infty} p^{n-1} + \Sigma{i=1}^{\infty} p^{n} $$

$$E(V) = 2 + \Sigma{i=1}^{\infty} 2 \ (n+1)\ p^{n} - \Sigma{i=1}^{\infty} 2 \ n \ p^{n} - 1 - \Sigma{i=1}^{\infty} p^{n} + \Sigma{i=1}^{\infty} p^{n} $$

$$E(V) = 1 + \Sigma_{i=1}^{\infty} 2 p^{n} $$

$$E(V) = 1 + \frac{2 \ p }{1-p}$$

Write a program to simulate the game and verify that your answer is correct.


    import numpy as np
    
    def expect(n,p):
        val = 0.
        for i in range(1,n+1):
            val += p**(i-1)*(1-p)*(2*i-1)
        return val
    
    def an(p):
        return 1+2*p/(1-p)
    
    def simulation(n,p):
        pays = []
        for i in range(n):
            i = 1
            while np.random.rand() < p:
                i += 1
            pays.append((2*i-1))
        pays = np.array(pays)
        return pays.mean()
        
    for p in np.linspace(0.1,0.9,10):
        print expect(1000,p), an(p), simulation(10000,p)
    


    1.22222222222 1.22222222222 1.2154
    1.46575342466 1.46575342466 1.4702
    1.76923076923 1.76923076923 1.7798
    2.15789473684 2.15789473684 2.1384
    2.67346938776 2.67346938776 2.6592
    3.39024390244 3.39024390244 3.3546
    4.45454545455 4.45454545455 4.3904
    6.2 6.2 6.186
    9.58823529412 9.58823529412 9.6414
    19.0 19.0 19.1224


## Linear Algebra Practice:
The stochastic matrix is central to the Markov process. It is a sqaure matrix specifying that probabilities of going from one state to the other such that every column of the matrix sums to 1.  The probability of entering a certain state depends only on the last state occupied and the stochastic matrix, not on any earlier states

Suppose that the 2004 **state of land use** in a city of 60 mi^2 of built-up
area is

```
In 2004:
   
C (Commercially Used): 25%
I (Industrially Used): 20%
R (Residentially Used): 55%
```

1. Find the **state of land use** in **2009** and **2014**,
   assuming that the transition probabilities for 5-year intervals are given
   by the matrix **A** and remain practically the same over the time considered.
   
   <div align="center">
      <img src="images/transition_matix_A.png">
   </div>
   
   


    import numpy as np
    
    A_5 = np.array([[0.7,0.1,0.0],[0.2,0.9,0.2],[0.1,0.0,0.8]])
    A_10 = A_5.dot(A_5)
    A_10
    
    print "2004: ", np.array([.25,0.20,0.55])
    print "2009: ", A_5.dot(np.array([.25,0.20,0.55]))
    print "2014: ", A_10.dot(np.array([.25,0.20,0.55]))

    2004:  [ 0.25  0.2   0.55]
    2009:  [ 0.195  0.34   0.465]
    2014:  [ 0.1705  0.438   0.3915]


###Part 1.2

This following question uses the `iris` dataset. Load the data in with the following code.
   
```python
from sklearn import datasets
# The 1st column is sepal length and the 2nd column is sepal width
sepalLength_sepalWidth = datasets.load_iris().data[:, :2]
```

1. Make a scatter plot of sepal width vs sepal length
  
2. Compute the mean vector (column-wise) of the data matrix. The `shape`
   of the mean vector should be `(1, 2)` and plot it.
 


    %matplotlib inline
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    sepalLength_sepalWidth = datasets.load_iris().data[:, :2]
    plt.plot(sepalLength_sepalWidth[:,0],sepalLength_sepalWidth[:,1],linewidth=0,marker='o',color='steelblue',alpha=0.25,label="Data")
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.plot(sepalLength_sepalWidth[:,0].mean(),sepalLength_sepalWidth[:,1].mean(),'x',
             markeredgewidth=10,markersize=20,color='indianred',label='Mean',alpha=0.5)
    print "Shape: ",sepalLength_sepalWidth.shape

    Shape:  (150, 2)



![png](http://www.bryantravissmith.com/img/GW03D1/output_5_1.png)


 3. Write a function (`euclidean_dist`) to calculate the euclidean distance
   between two **column vectors (not row vector)**. Your function should check
   if the vectors are column vectors and the shape of the two vectors are the same .


    def euclidean_dist(c1,c2):
        if c2.shape==c1.shape and c1.shape[1]==1:
            return np.sqrt(np.sum(np.power(c1-c2,2)))
        return None
    
    a = np.ones((10,1))
    b = np.zeros((10,1))
    euclidean_dist(a,b)==np.sqrt(10)




    True



4. Write a function (`cosine_sim`) to calculate the cosine similarity_between 
   two **column vectors (not row vector)**.
   


    def cosine_sim(c1,c2):
        if c2.shape==c1.shape and c1.shape[1]==1:
            result = c1.transpose().dot(c2)/(np.linalg.norm(c1)*np.linalg.norm(c2))
            return result[0,0]
        return None
    
    a = np.array([1,0]).reshape(2,1)
    b = np.array([0,1]).reshape(2,1)
    print cosine_sim(a,b)==0
    a = np.array([1,0]).reshape(2,1)
    b = np.array([1,1]).reshape(2,1)
    print cosine_sim(a,b) == 1/np.sqrt(2)

    True
    True


5. Write a function that would loop through all the data points in a given matrix and 
   calculate the given distance metric between each of the data point and the mean
   vector. Use the function to compute Euclidean Distance and Cosine Similarity between each of
   the data points and the mean of the data points. You should be able to call the function
   in this manner:

6. Plot histograms of the euclidean distances and cosine similarities.
   


    def compute_dist(data,func):
        means = data.mean(axis=0).transpose().reshape(data.shape[1],1)
        dists = []
        for i in range(data.shape[0]):
            dists.append(func(data[i,:].reshape(data.shape[1],1),means))
        dists = np.array(dists)
        dists = dists.reshape(data.shape[0],1)
        return dists
            
    values = compute_dist(sepalLength_sepalWidth,euclidean_dist)
    plt.hist(values[:,0],color='steelblue',alpha=0.5,edgecolor='black',linewidth=0.1)
    plt.xlabel('Euclidean Distance From Mean')
    plt.ylabel('Count')
    plt.show()
    values = compute_dist(sepalLength_sepalWidth,cosine_sim)
    plt.hist(values[:,0],color='steelblue',alpha=0.5,linewidth=0.1)
    plt.xlabel('Cosign Similarity From Mean')
    plt.ylabel('Count')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_11_0.png)



![png](http://www.bryantravissmith.com/img/GW03D1/output_11_1.png)


## Extra Credit: Implementing the PageRank Algorithm

The [Page Rank Algorithm](http://en.wikipedia.org/wiki/PageRank) is used by Google
Search (in their early days) to rank websites in their search engine in terms 
of the importance of webpages. 
[More about PageRank](http://books.google.com/books/p/princeton?id=5o_K4rri1CsC&printsec=frontcover&source=gbs_ViewAPI&hl=en#v=onepage&q&f=false)

We will implement PageRank on this simple network of websites.

   <div align="center">
    <img src="images/pageweb.png">
   </div>

**In the above image:**
   - Each node is a web page
   - Each directed edge corresponds to one page referencing the other
   - These web pages correspond to the states our Markov chain can be in
   - Assume that the model of our chain is that of a random surfer/walker.

In this model, we transition from one web page (state) to the next with
equal probability (to begin).  Or rather we randomly pick an outgoing edge
from our current state.  Before we can do any sort of calculation we need to
know how we will move on this Markov Chain.


    PR = np.array([[0,1,0,0,0],[0.5,0,0.5,0,0],[0.333333,0.333333,0,0,0.333333],[1.,0,0,0,0],[0,0.3333333,0.3333333,0.3333333,0]]).transpose()
    PR




    array([[ 0.       ,  0.5      ,  0.333333 ,  1.       ,  0.       ],
           [ 1.       ,  0.       ,  0.333333 ,  0.       ,  0.3333333],
           [ 0.       ,  0.5      ,  0.       ,  0.       ,  0.3333333],
           [ 0.       ,  0.       ,  0.       ,  0.       ,  0.3333333],
           [ 0.       ,  0.       ,  0.333333 ,  0.       ,  0.       ]])



2. Now that we have a transition matrix, the next step is to iterate on this
   from one page to the next (like someone blindly navigating the internet) and
   see where we end up. The probability distribution for our random surfer can
   be described in this matrix notation as well (or vector rather).

   Initialize a vector for the probability of where our random surfer is.
   It will be a vector with length equal to the number of pages.
   Initialize it to be equally probable to start on any page
   (i.e. you start randomly in a state on the chain).


    start = np.array([0.2,0.2,0.2,0.2,0.2]).reshape(5,1)
    start




    array([[ 0.2],
           [ 0.2],
           [ 0.2],
           [ 0.2],
           [ 0.2]])



3. To take a step on the chain, simply matrix multiple our user vector by the
   transition matrix.
   After one iteration, what is the most likely location for your random surfer?


    step1 = PR.dot(start)
    step1




    array([[ 0.3666666 ],
           [ 0.33333326],
           [ 0.16666666],
           [ 0.06666666],
           [ 0.0666666 ]])



4. Plot how the probabilities change.
   Iterate the matrix through the first ten steps.
   At each step create a bar plot of the surfers probability vector.


    steps = start.copy()
    plt.figure(figsize=(15,8))
    for i in range(10):
        steps = PR.dot(steps)
        ax = plt.subplot(2,5,i+1)
        plt.bar(np.arange(5),steps[:,0],color='steelblue',alpha=0.5,linewidth=0.1)
        ax.set_xticklabels (('A', 'B', 'C', 'D', 'E') )
        plt.title(str(i+1) + " Steps")
    plt.show()
    print "Final Distribution: ",steps


![png](http://www.bryantravissmith.com/img/GW03D1/output_19_0.png)


    Final Distribution:  [[ 0.29401788]
     [ 0.38811497]
     [ 0.22055748]
     [ 0.02429332]
     [ 0.07301416]]


5. This time to compute the stationary distribution, we can use numpy's
   matrix operations. Using the function for calculating [eigenvectors](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) compute the
   stationary distibution (page rank).  Is it the same as what you found
   from above?  What is it's eigenvalue?


    values,Vectors = np.linalg.eig(PR)
    values




    array([ 0.99999977+0.j        , -0.64800046+0.27072373j,
           -0.64800046-0.27072373j,  0.14800058+0.30123034j,
            0.14800058-0.30123034j])




    solution = np.real(Vectors[:,0]/np.sum(Vectors[:,0]))
    solution.reshape(5,1)




    array([[ 0.29268293],
           [ 0.39024392],
           [ 0.21951223],
           [ 0.02439023],
           [ 0.07317069]])




    import pandas as pd
    df = pd.DataFrame(columns=["A","B","C","D","E"])
    steps = start.copy()
    for i in range(20):
        df.loc[i] = steps[:,0].transpose()
        steps = PR.dot(steps)
    
    df.plot()
    plt.legend(bbox_to_anchor=(0.08,1),ncol=5, loc=3, borderaxespad=0.)
    print "Final Distribution: "
    steps

    Final Distribution: 





    array([[ 0.29265924],
           [ 0.39030474],
           [ 0.21945581],
           [ 0.02438301],
           [ 0.07319274]])




![png](http://www.bryantravissmith.com/img/GW03D1/output_23_2.png)


We can see that the stationary state is found in 6 states.   The vector produced by this method matches the eigenvectors found by numpy linear algrebra library


    PR.dot(steps)




    array([[ 0.29268725],
           [ 0.39020868],
           [ 0.21954995],
           [ 0.02439758],
           [ 0.07315186]])



## Exploratory Data Analysis (EDA)

Exploratory data analysis is a first crucial step to building predictive models from your data. EDA allows you
to confirm or invalidate some of the assumptions you are making about your data and understand relationships between your variables.
 
<br>
 
In this scenario, you are a data scientist at [Bay Area Bike Share](http://www.bayareabikeshare.com/). Your task
is to provide insights on bike user activity and behavior to the products team. 


1. Load the file `data/201402_trip_data.csv` into a dataframe.


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as scs
    import statsmodels.api as sm
    from pandas.tools.plotting import scatter_matrix
    %matplotlib inline
    
    df = pd.read_csv('data/201402_trip_data.csv')
    
    df['start_date'] = pd.to_datetime(df.start_date)
    df['end_date'] = pd.to_datetime(df.end_date)
    df['month'] = df.start_date.dt.month
    df['dayofweek'] = df.start_date.dt.dayofweek
    df['date'] = df.start_date.dt.date
    df['hour'] = df.start_date.dt.hour


    df.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 144015 entries, 0 to 144014
    Data columns (total 15 columns):
    trip_id              144015 non-null int64
    duration             144015 non-null int64
    start_date           144015 non-null datetime64[ns]
    start_station        144015 non-null object
    start_terminal       144015 non-null int64
    end_date             144015 non-null datetime64[ns]
    end_station          144015 non-null object
    end_terminal         144015 non-null int64
    bike_#               144015 non-null int64
    subscription_type    144015 non-null object
    zip_code             137885 non-null object
    month                144015 non-null int64
    dayofweek            144015 non-null int64
    date                 144015 non-null object
    hour                 144015 non-null int64
    dtypes: datetime64[ns](2), int64(8), object(5)
    memory usage: 17.6+ MB


2. Group the bike rides by `month` and count the number of users per month. Plot the number of users for each month. 
   What do you observe? Provide a likely explanation to your observation. Real life data can often be messy/incomplete
   and cursory EDA is often able to reveal that.
   


    m = df.groupby('month')
    a = m.trip_id.count().tolist()
    a = a[2:]+a[:2]
    fig,ax = plt.subplots()
    plt.bar(range(7),a,color='steelblue',alpha=0.5)
    ax.set_xticklabels(('Aug','Sep','Oct',"Nov","Dec",'Jan','Feb'))
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_30_0.png)


We see that we do not have data for the entire month of august, so it is not fair to evaluate it to the other months.  We see that there is a slowdown in december, and febuary has less days.   It might make more sense to look at the daily usage rates. 

3. Plot the daily user count from September to December. Mark the `mean` and `mean +/- 1.5 * Standard Deviation` as 
   horizontal lines on the plot. This would help you identify the outliers in your data. Describe your observations. 
   


    duc = df[df.month.isin([9, 10, 11, 12])].groupby('date')
    counts = duc.trip_id.count().reset_index().set_index('date')
    counts.columns = ['Count']
    high = counts.mean()+1.5*counts.std()
    low = counts.mean()-1.5*counts.std()
    counts.plot(color='steelblue',lw=2,alpha=.75)
    plt.axhline(high.values,color='black',linestyle='--')
    plt.axhline(low.values,color='black',linestyle='--')
    plt.xticks(rotation=30)
    plt.ylabel('Daily User Count')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_33_0.png)


The dashed black lines are the 'outlier' boundaries.   Though these are not the traditional definition of an outlire, we see points outside of these bounds.  The issues is that we have seasonal variation.  The low months pull down the average making high months look like possible outliers.   I do not believe we see any outlires in this data set, and won't believe it until I understand the cyclical nature of the data.

4. Plot the distribution of the daily user counts for all months as a histogram. Fit a 
   [KDE](http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html) to the histogram.
   What is the distribution and explain why the distribution might be shaped as such. 
    


    duc = df.groupby('date')
    counts = duc.trip_id.count()
    counts.hist(alpha=.5, linewidth=0, bins=20, normed=True)
    
    kde = scs.gaussian_kde(counts)
    s = np.linspace(counts.min(), counts.max(), 100)
    plt.plot(s, kde(s), color='indianred', lw=3)




    [<matplotlib.lines.Line2D at 0x11123e610>]




![png](http://www.bryantravissmith.com/img/GW03D1/output_36_1.png)


This looks like we could have two different contributers to daily user counts.   One that has a mean of around 1000, and another with a mean around 400.   The combination of the two sources could produce a double peaked distribution like this.  


    plt.figure(figsize=(14,8))
    ax = plt.gca()
    df['weekend'] = df.dayofweek.isin([5,6])
    duc_we = df[df['weekend']].groupby('date')
    duc_wd = df[~df['weekend']].groupby('date')
    counts_we = duc_we.trip_id.count().reset_index().set_index('date')
    counts_wd = duc_wd.trip_id.count().reset_index().set_index('date')
    counts_we.hist(alpha=.7, linewidth=0, bins=20, normed=True, color='burlywood',ax=ax)
    counts_wd.hist(alpha=.7, linewidth=0, bins=20, normed=True, color='seagreen',ax=ax)
    
    kde_we = scs.gaussian_kde(counts_we.values[:,0])
    kde_wd = scs.gaussian_kde(counts_wd.values[:,0])
    plt.plot(s, kde_we(s), color='goldenrod', lw=3, label='Weekend')
    plt.plot(s, kde_wd(s), color='forestgreen', lw=3, label='Weekday')
    plt.xlabel('Number of Daily Users')
    plt.ylabel('Probability Density')
    plt.title('Weekend vs Weekday Daily User Count')
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_38_0.png)


We can see that from this distirbution that the weekend users seem to be around 400, and the weekday user counts seem to be peaked around 1000.  The combintation of the two distributions produces the initial histogram we looked at

5. Now we are going to explore hourly trends of user activity. Group the bike rides by `date` and `hour` and count 
   the number of rides in the given hour on the given date. Make a 
   [boxplot](http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/) of the hours in the day **(x)** against
   the number of users **(y)** in that given hour. 
   
   


    plt.figure(figsize=(14,8))
    ax=plt.gca()
    huc = df.groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Hourly User Count')
    plt.show()
    
    huc.trip_id.count().reset_index().boxplot('trip_id',by='hour')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_41_0.png)



![png](http://www.bryantravissmith.com/img/GW03D1/output_41_1.png)


7. Replot the boxplot from above after binning your data into weekday and weekend. Describe the differences you observe between hour user activity between weekday and weekend? 
     


    plt.figure(figsize=(14,6))
    ax=plt.subplot(1,2,1)
    huc = df[df.weekend].groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekend Hourly User Count')
    plt.title('Weekend Hourly User Counts')
    
    ax=plt.subplot(1,2,2)
    huc = df[~df.weekend].groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekday Hourly User Count')
    plt.title('Weekday Hourly User Counts')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_43_0.png)


Weekend users center in daylight hours and peak shortly after lunch.   Weekday users seem to be commuters, using the bikes before and after work hours, with a similar proportion of users using the bikes around lunchtime as weekend users.

    
8. There are two types of bike users (specified by column `Subscription Type`: `Subscriber` and `Customer`. Given this
   information and the weekend and weekday categorization, plot and inspect the user activity trends. Suppose the 
   product team wants to run a promotional campaign, what are you suggestions in terms of who the promotion should 
   apply to and when it should apply for the campaign to be effective?
     


    plt.figure(figsize=(14,12))
    ax=plt.subplot(2,2,1)
    huc = df[(df.weekend)&(df.subscription_type=='Customer')].groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekend Hourly User Count')
    plt.ylim([0,150])
    plt.title('Customer Weekend Hourly User Counts')
    
    ax=plt.subplot(2,2,2)
    huc = df[(~df.weekend)&(df.subscription_type=='Customer')].groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekday Hourly User Count')
    plt.ylim([0,150])
    plt.title('Customer Weekday Hourly User Counts')
    
    ax=plt.subplot(2,2,3)
    huc = df[(df.weekend)&(df.subscription_type=='Subscriber')].groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekend Hourly User Count')
    plt.ylim([0,150])
    plt.title('Subscriber Weekend Hourly User Counts')
    
    ax=plt.subplot(2,2,4)
    huc = df[(~df.weekend)&(df.subscription_type=='Subscriber')].groupby(['date','hour'])
    huc_vals = huc.trip_id.count().unstack()
    huc_vals.fillna(0.0,inplace=True)
    huc_vals.plot(kind='box',ax=ax)
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekday Hourly User Count')
    plt.ylim([0,150])
    plt.title('Subscriber Weekday Hourly User Counts')
    
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_46_0.png)


It seems that subscribers are mostly commuters, and responsible for the weekday double peek.  Because there seems to be a large number of bikes not be used on the weekends, or during lunch periods, a promotion targeting customers during non-peak times could help bring in more revenue.  

## Linear Regression

Linear regression is an approach to modeling the relationship between a continuous dependent (**y**) variable and 
one or more continuous independent (**x**) variables. Here you will be introduced to fitting the model and interpreting
the results before we dive more into the details of linear regression tomorrow.

1. We will be using the `prestige` data in `statsmodels`. `statsmodels` is the de facto library for performing regression tasks in Python. Load the data with the follow code.


    import statsmodels.api as sm
    prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
    y = prestige['prestige']
    x = prestige[['income', 'education']].astype(float)





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>education</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accountant</th>
      <td>62</td>
      <td>86</td>
    </tr>
    <tr>
      <th>pilot</th>
      <td>72</td>
      <td>76</td>
    </tr>
    <tr>
      <th>architect</th>
      <td>75</td>
      <td>92</td>
    </tr>
    <tr>
      <th>author</th>
      <td>55</td>
      <td>90</td>
    </tr>
    <tr>
      <th>chemist</th>
      <td>64</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>



2. Explore the data by making a [scatter_matrix](http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html#visualization-scatter-matrix)
   and a [boxplot](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html)
   to show the range of each of your variables.
   
   


    scatter_matrix(prestige[['prestige','income','education']], diagonal='kde', figsize=(10,10))
    plt.show()
    
    prestige[['prestige','income','education']].boxplot()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D1/output_51_0.png)



![png](http://www.bryantravissmith.com/img/GW03D1/output_51_1.png)


It seems that the values have been normalized between 0 and 100.  It also looks that prestige is related to education and income, but so are income and education.   Since these values are not independant, we might see funny results in our fits.

3. The beta coefficients of a linear regression model can be calculated by solving the normal equation.
   Using numpy, write a function that solves the **normal equation** (below).
   As input your function should take a matrix of features (**x**) and
   a vector of target (**y**). You should return a vector of beta coefficients 
   that represent the line of best fit which minimizes the residual. 
   Calculate  R<sup>2</sup>. 
   
   <div align="center">
      <img height="30" src="images/normal_equation.png">
   </div>


    x = sm.add_constant(x)
    x.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>income</th>
      <th>education</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accountant</th>
      <td>1</td>
      <td>62</td>
      <td>86</td>
    </tr>
    <tr>
      <th>pilot</th>
      <td>1</td>
      <td>72</td>
      <td>76</td>
    </tr>
    <tr>
      <th>architect</th>
      <td>1</td>
      <td>75</td>
      <td>92</td>
    </tr>
    <tr>
      <th>author</th>
      <td>1</td>
      <td>55</td>
      <td>90</td>
    </tr>
    <tr>
      <th>chemist</th>
      <td>1</td>
      <td>64</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>




    def solve_normal(matrix, vector):
        x = matrix.values
        y = vector.values
        return np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    
    betas = solve_normal(x,y)
    print "Constant: ", betas[0]
    print "Income: ", betas[1]
    print "Education: ", betas[2]

    Constant:  -6.0646629221
    Income:  0.598732821529
    Education:  0.545833909401



    TSS = np.sum(np.power(y.values-y.values.mean(),2))
    RSS = np.sum(np.power(y.values-np.dot(x.values,betas),2))
    print "R-Squared: ", 1-RSS/TSS

    R-Squared:  0.828173417254


3. Verify your results using statsmodels. Use the code below as a reference.


    import statsmodels.api as sm
    model = sm.OLS(y, x).fit()
    model.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>prestige</td>     <th>  R-squared:         </th> <td>   0.828</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.820</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   101.2</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 16 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>8.65e-17</td>
</tr>
<tr>
  <th>Time:</th>                 <td>07:23:19</td>     <th>  Log-Likelihood:    </th> <td> -178.98</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    45</td>      <th>  AIC:               </th> <td>   364.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    42</td>      <th>  BIC:               </th> <td>   369.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>     <td>   -6.0647</td> <td>    4.272</td> <td>   -1.420</td> <td> 0.163</td> <td>  -14.686     2.556</td>
</tr>
<tr>
  <th>income</th>    <td>    0.5987</td> <td>    0.120</td> <td>    5.003</td> <td> 0.000</td> <td>    0.357     0.840</td>
</tr>
<tr>
  <th>education</th> <td>    0.5458</td> <td>    0.098</td> <td>    5.555</td> <td> 0.000</td> <td>    0.348     0.744</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.279</td> <th>  Durbin-Watson:     </th> <td>   1.458</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.528</td> <th>  Jarque-Bera (JB):  </th> <td>   0.520</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.155</td> <th>  Prob(JB):          </th> <td>   0.771</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.426</td> <th>  Cond. No.          </th> <td>    163.</td>
</tr>
</table>



4. Interpret your result summary, focusing on the beta coefficents, p-values, F-statistic, and the R<sup>2</sup>. 

The results from the manual calculation match the values from statsmodels' OLS regression.   We have 82.8% of the varience is explained by the model.  The F-statistics, assuming independant variables, says our model does describe some of the behavior of the prestige.   The coefficients say that for every unit increase in income, we have a 0.5987 incraese in prestige, and for every unit increase in educaiton we have a 0.5458 increase in prestigue.  The issue with the model, highlighted by the p-value on the constant term, is that as educaiton increases, so does income.   They will increase together, somehow double increasing the prestigue.   We need a way to remove this co-linearity between income and educaiton


    x = np.random.rand(20,1)*10
    y = 5*x+2+np.random.rand(20,1)*30
    z = 2*x+4*y+2+np.random.rand(20,1)*40
    ones = np.ones(20).reshape(20,1)
    
    X = pd.DataFrame({'x':x[:,0],'y':y[:,0],'c':ones[:,0]})
    Y = pd.DataFrame({'z':z[:,0]})
    model = sm.OLS(Y, X).fit()
    model.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>z</td>        <th>  R-squared:         </th> <td>   0.975</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.972</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   325.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 16 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>2.79e-14</td>
</tr>
<tr>
  <th>Time:</th>                 <td>07:40:32</td>     <th>  Log-Likelihood:    </th> <td> -78.012</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    20</td>      <th>  AIC:               </th> <td>   162.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    17</td>      <th>  BIC:               </th> <td>   165.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
  <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>c</th> <td>   18.9688</td> <td>    9.231</td> <td>    2.055</td> <td> 0.056</td> <td>   -0.507    38.445</td>
</tr>
<tr>
  <th>x</th> <td>    2.2866</td> <td>    2.065</td> <td>    1.107</td> <td> 0.284</td> <td>   -2.071     6.644</td>
</tr>
<tr>
  <th>y</th> <td>    4.0274</td> <td>    0.378</td> <td>   10.667</td> <td> 0.000</td> <td>    3.231     4.824</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.159</td> <th>  Durbin-Watson:     </th> <td>   1.527</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.206</td> <th>  Jarque-Bera (JB):  </th> <td>   1.277</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.090</td> <th>  Prob(JB):          </th> <td>   0.528</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.775</td> <th>  Cond. No.          </th> <td>    158.</td>
</tr>
</table>



We can see that if we take away the constant, removing a degree of freedom, the R-squared becomes larger.   I think this could be a sign that we have a covariance in the variables. 


    
