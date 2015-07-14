
#Galvanize Immersive Data Science

##Week 6 - Day 3

I liked our quiz today.  It was more SQL, but we had to load data into a local PostgreSQL database and test the queries.  We were give two table discriptions:

```
log
    userid
    tmstmp
    itemid
    event

items
    itemid
    name
    category
```

Where the possible events are:
* `view`: Viewing the details of an item
* `add`: Adding the item to shopping cart
* `buy`: Buying an item

We had to write two queries.  One that would select all the user, item pairs where a user has added an item to the cart but has not bought it.  If they bought that item, then added it to their cart again, that add should be counted.  



		SELECT a.userid, a.itemid FROM log a 
		JOIN (SELECT userid, 
						itemid, 
						MAX(tmstmp) as last_tmstmp 
			FROM log
			GROUP BY userid, itemid) b 
		ON a.userid = b.userid
		AND a.itemid = b.itemid
		AND a.event = 'add'
		AND a.tmstmp = b.last_tmstamp

The next query we had to write was to fetch the ratio of views to purchases for each category.   I decided to get the count for views and joins for each item, then group by category for the ratio.  

		SELECT i.category,
			SUM( a.buy_count ) / SUM( a.view_count ) 
				as ration
		FROM items i JOIN (
			SELECT itemid,
			SUM( CASE(event = 'buy' THEN 1 ELSE 0) )
				as buy_count,
			SUM( CASE(event = 'view' THEN 1 ELSE 0) )
				as view_count 
			FROM log 
			GROUP BY itemid) a
		ON i.itemid = a.itemid
		GROUP BY i.category

##Collaborative Filtering

Collaborative Filtering is a way to make recommendations based on using related data to make predictions.  There are item-item collaborative filtering techniques that take interest in one item to be a signal that you might be interested in a similar item.  User-user collaborative filtering takes users that are similar to you and recommends things these users have liked that you have not tried/bought/experienced.   

Our morning sprint is making an item-item collaborative filtering enging for moviews in the data from 
You will be using the [MovieLens](http://grouplens.org/datasets/movielens/)

Our process for making recommendations will involved the following: 

1.  Calculated how similarly movies are related using cosine similarity  
2.  Construct a neighborhood for each movie of the N most similar movies  
3.  Take the weighted average of each user's previous ratings for each movie's neighborhood
4.  Return prediction

Our ItemItemRecommender Class is below.  This class does not take advantage of scipy's sparse matrixes.  The reason is I found that for this dataset that it ran measurably slower compared to using numpy matrixes.   I am not sure if I am just unfarmiliar with the best practices of sparse matrixes, or if there is something characteristic of sparse matrixes.  


    import numpy as np
    class ItemItemRecommender(object):
        
        def __init__(self,neighborhood_size=8):
            """Initializes the parameters of the model.
            """
            self.neighborhood_size = neighborhood_size
    
        def fit(self,matrix):
            """Implements the model and fits it to the data passed as an
            argument.
    
            Stores objects for describing model fit as class attributes.
            """
            self.matrix = matrix
            self._set_neighborhoods()
    
        def _set_neighborhoods(self):
            """Gets the items most similar to each other item.
    
            Should set a class attribute with a matrix that is has
            number of rows equal to number of items and number of 
            columns equal to neighborhood size. Entries of this matrix
            will be indexes of other items.
    
            You will call this in your fit method.
            """
            from sklearn.metrics.pairwise import cosine_similarity
            self.sim = cosine_similarity(self.matrix.T)
            self.neighborhoods = np.argsort(self.sim[:,:],1)[:,-self.neighborhood_size:]
    
        def pred_one_user(self,index):
            """Accept user id as arg. Return the predictions for a single user.
            
            Optional argument to specify whether or not timing should be provided
            on this operation.
            """
            scores = self.matrix[index,:].nonzero()[0]
            predictions = np.zeros(self.matrix.shape[1])
            for i in range(self.matrix.shape[1]):
                relevant_items = np.intersect1d(scores,self.neighborhoods[i],assume_unique=True)
                #if i in scores:
                #    result = self.matrix[mask,:][0,i]
                #else:
                result = np.sum(self.matrix[index,relevant_items]*self.sim[i,relevant_items])
                if result > 0:
                    result = result/np.sum(self.sim[i,relevant_items])
                predictions[i] = result
            return predictions
            
        def pred_all_users(self):
            """Repeated calls of pred_one_user, are combined into a single matrix.
            Return value is matrix of users (rows) items (columns) and predicted
            ratings (values).
    
            Optional argument to specify whether or not timing should be provided
            on this operation.
            """
            predictions = np.zeros(self.matrix.shape)
            for i in range(self.matrix.shape[0]):
                predictions[i] = self.pred_one_user(i)
            return predictions
            
            
        def top_n_recs(self,index,n):
            """Takes user_id argument and number argument.
    
            Returns that number of items with the highest predicted ratings,
            after removing items that user has already rated.
            """
            scores = self.matrix[index,:].nonzero()[0]
            predictions = self.pred_one_user(index)
            predictions[scores] = -1
            return np.argsort(predictions)[::-1][:n]


We are ready to import our data


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from ItemRecommender import ItemItemRecommender
    %matplotlib inline
    ratings_contents = pd.read_table("../data/u.data",
                                         names=["user", "movie", "rating", "timestamp"])
    df = ratings_contents.drop('timestamp',axis=1).set_index(['user','movie']).unstack()
    df.columns = df.columns.droplevel()
    df = df.fillna(0)
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movie</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1682 columns</p>
</div>




    from time import time
    IIR = ItemItemRecommender(20)
    t1 = time()
    IIR.fit(df.values)
    print "Time to Fit: ", time()-t1

    Time to Fit:  0.853658914566


The item-item recommender is finding a cosine similarity matrix, which is relatively quick.  Predictions, however, will not that quick.


    t1=time()
    print "Prediction: ", np.round(IIR.pred_one_user(0)[:10],0)
    print "Actual:     ", df.values[0,:10]
    print "Time to Predict 1 User: ", time()-t1

    Prediction:  [ 4.  3.  4.  4.  3.  5.  4.  4.  4.  4.]
    Actual:      [ 5.  3.  4.  3.  3.  5.  4.  1.  5.  3.]
    Time to Predict 1 User:  0.0642509460449


If we want to predict for all users.


    t1 = time()
    predictions = IIR.pred_all_users()
    print "Time to predict 943 Users:", time()-t1

    Time to predict 943 Users: 23.9097139835


We can also increase the size of the neighborhoods for similar movies.  We just did 20, lets change it to 75.


    from time import time
    IIR = ItemItemRecommender(75)
    t1 = time()
    IIR.fit(df.values)
    print "Time to Fit: ", time()-t1
    t1=time()
    print "Prediction: ", np.round(IIR.pred_one_user(0)[:10],0)
    print "Actual:     ", df.values[0,:10]
    print "Time to Predict 1 User: ", time()-t1
    t1 = time()
    predictions = IIR.pred_all_users()
    print "Time to predict 943 Users:", time()-t1

    Time to Fit:  0.224536895752
    Prediction:  [ 4.  4.  4.  4.  4.  5.  4.  4.  4.  4.]
    Actual:      [ 5.  3.  4.  3.  3.  5.  4.  1.  5.  3.]
    Time to Predict 1 User:  0.0480990409851
    Time to predict 943 Users: 32.1997280121


We can see that the neighborhood size is going to change the results.  One thing I know is that there are many more action moviews then documentaries in the dataset.   This does not allow us to filter based on the different number of each.   

The larger neighborhood also increases the computation size.  Papers have published results that it does reduce the mean square error.  For a production item-item recommender like the one we built, we would have to update our predictions at a regualar basis.  This would have to be off line because of how long it takes to fit.

##Top Movies

We also have movie details for the data in our recommender.  I am going to import that data and see what movies are being recommended.


    idf = pd.read_table("../data/u.item",sep="|",header=None)
    idf = idf.iloc[:,[1,2]]
    idf.columns = ['Title','Date']
    idf.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GoldenEye (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Four Rooms (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Get Shorty (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Copycat (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
  </tbody>
</table>
</div>



The first user rated the following movies:


    user1_rating_indexes = df.values[0,:].nonzero()[0]
    user1 = idf.loc[user1_rating_indexes,:]
    user1['Ratings'] = df.values[0,user1_rating_indexes]
    user1[user1.Ratings==5]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Date</th>
      <th>Ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story (1995)</td>
      <td>01-Jan-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Shanghai Triad (Yao a yao yao dao waipo qiao) ...</td>
      <td>01-Jan-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dead Man Walking (1995)</td>
      <td>01-Jan-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Usual Suspects, The (1995)</td>
      <td>14-Aug-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Mighty Aphrodite (1995)</td>
      <td>30-Oct-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Postino, Il (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Mr. Holland's Opus (1995)</td>
      <td>29-Jan-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>French Twist (Gazon maudit) (1995)</td>
      <td>01-Jan-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Antonia's Line (1995)</td>
      <td>01-Jan-1995</td>
      <td>5</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Crumb (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Clerks (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Dolores Claiborne (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Eat Drink Man Woman (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Hoop Dreams (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Star Wars (1977)</td>
      <td>01-Jan-1977</td>
      <td>5</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Professional, The (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Priest (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Three Colors: Red (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Three Colors: Blue (1993)</td>
      <td>01-Jan-1993</td>
      <td>5</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Shawshank Redemption, The (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Hudsucker Proxy, The (1994)</td>
      <td>01-Jan-1994</td>
      <td>5</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Jurassic Park (1993)</td>
      <td>01-Jan-1993</td>
      <td>5</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Remains of the Day, The (1993)</td>
      <td>01-Jan-1993</td>
      <td>5</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Searching for Bobby Fischer (1993)</td>
      <td>01-Jan-1993</td>
      <td>5</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Blade Runner (1982)</td>
      <td>01-Jan-1982</td>
      <td>5</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Nightmare Before Christmas, The (1993)</td>
      <td>01-Jan-1993</td>
      <td>5</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Welcome to the Dollhouse (1995)</td>
      <td>24-May-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Terminator 2: Judgment Day (1991)</td>
      <td>01-Jan-1991</td>
      <td>5</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Fargo (1996)</td>
      <td>14-Feb-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Kids in the Hall: Brain Candy (1996)</td>
      <td>12-Apr-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Raiders of the Lost Ark (1981)</td>
      <td>01-Jan-1981</td>
      <td>5</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Brazil (1985)</td>
      <td>01-Jan-1985</td>
      <td>5</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Aliens (1986)</td>
      <td>01-Jan-1986</td>
      <td>5</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Good, The Bad and The Ugly, The (1966)</td>
      <td>01-Jan-1966</td>
      <td>5</td>
    </tr>
    <tr>
      <th>177</th>
      <td>12 Angry Men (1957)</td>
      <td>01-Jan-1957</td>
      <td>5</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Return of the Jedi (1983)</td>
      <td>14-Mar-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>182</th>
      <td>Alien (1979)</td>
      <td>01-Jan-1979</td>
      <td>5</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Henry V (1989)</td>
      <td>01-Jan-1989</td>
      <td>5</td>
    </tr>
    <tr>
      <th>190</th>
      <td>Amadeus (1984)</td>
      <td>01-Jan-1984</td>
      <td>5</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Terminator, The (1984)</td>
      <td>01-Jan-1984</td>
      <td>5</td>
    </tr>
    <tr>
      <th>195</th>
      <td>Dead Poets Society (1989)</td>
      <td>01-Jan-1989</td>
      <td>5</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Graduate, The (1967)</td>
      <td>01-Jan-1967</td>
      <td>5</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Nikita (La Femme Nikita) (1990)</td>
      <td>01-Jan-1990</td>
      <td>5</td>
    </tr>
    <tr>
      <th>201</th>
      <td>Groundhog Day (1993)</td>
      <td>01-Jan-1993</td>
      <td>5</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Back to the Future (1985)</td>
      <td>01-Jan-1985</td>
      <td>5</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Cyrano de Bergerac (1990)</td>
      <td>01-Jan-1990</td>
      <td>5</td>
    </tr>
    <tr>
      <th>207</th>
      <td>Young Frankenstein (1974)</td>
      <td>01-Jan-1974</td>
      <td>5</td>
    </tr>
    <tr>
      <th>215</th>
      <td>When Harry Met Sally... (1989)</td>
      <td>01-Jan-1989</td>
      <td>5</td>
    </tr>
    <tr>
      <th>220</th>
      <td>Breaking the Waves (1996)</td>
      <td>15-Nov-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>222</th>
      <td>Sling Blade (1996)</td>
      <td>22-Nov-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>223</th>
      <td>Ridicule (1996)</td>
      <td>27-Nov-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>227</th>
      <td>Star Trek: The Wrath of Khan (1982)</td>
      <td>01-Jan-1982</td>
      <td>5</td>
    </tr>
    <tr>
      <th>234</th>
      <td>Mars Attacks! (1996)</td>
      <td>13-Dec-1996</td>
      <td>5</td>
    </tr>
    <tr>
      <th>241</th>
      <td>Kolya (1996)</td>
      <td>24-Jan-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>245</th>
      <td>Chasing Amy (1997)</td>
      <td>01-Jan-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>252</th>
      <td>Pillow Book, The (1995)</td>
      <td>13-Jun-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Contact (1997)</td>
      <td>11-Jul-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>267</th>
      <td>Chasing Amy (1997)</td>
      <td>01-Jan-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>268</th>
      <td>Full Monty, The (1997)</td>
      <td>01-Jan-1997</td>
      <td>5</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Gattaca (1997)</td>
      <td>01-Jan-1997</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 3 columns</p>
</div>



Based on this list, our recommener is making recommendations for movies that user has not watched.  The Top 10 Movies recommended to this user are:


    idf.loc[IIR.top_n_recs(0,10),:]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>810</th>
      <td>Thirty-Two Short Films About Glenn Gould (1993)</td>
      <td>01-Jan-1993</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>Spirits of the Dead (Tre passi nel delirio) (1...</td>
      <td>01-Jan-1968</td>
    </tr>
    <tr>
      <th>1526</th>
      <td>Senseless (1998)</td>
      <td>09-Jan-1998</td>
    </tr>
    <tr>
      <th>1462</th>
      <td>Boys, Les (1997)</td>
      <td>01-Jan-1997</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Crooklyn (1994)</td>
      <td>01-Jan-1994</td>
    </tr>
    <tr>
      <th>1544</th>
      <td>Frankie Starlight (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>Mrs. Dalloway (1997)</td>
      <td>01-Jan-1997</td>
    </tr>
    <tr>
      <th>781</th>
      <td>Little Odessa (1994)</td>
      <td>01-Jan-1994</td>
    </tr>
    <tr>
      <th>1369</th>
      <td>I Can't Sleep (J'ai pas sommeil) (1994)</td>
      <td>01-Jan-1994</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>Search for One-eye Jimmy, The (1996)</td>
      <td>01-Jan-1996</td>
    </tr>
  </tbody>
</table>
</div>



Unfortunately we do not have a measure for how successful these recommnedations are.  We can not have the user watch them, and if we did we would need the user to follow up with a rating.  These are all predicted to be 5 star movies for this user.

One of the shortcomings I see from looking up some of these titles on IMDB is the measure of similarity is only user ratings.   This metric does not deal with the property of the movies.  I, personally, like Kevin Spacey.   He has been in a wide variety of movies, and I have enjoyed movies that are out of my traditional preferences in part of his performances.   That is not weighted into our similarity.  Including it, however, will also increase the computational time for updates.  

## Matrix Factorization Based Recommenders

Our afternoon paired sprint was implementing matrix factorization based recommenders.   These involved constructing two matrixes that multiply together to produce the rating matrix.  The idea is that through the user of latent variables, the matrix product will make predictions for movies that are not yet reviewed.  

We were introduced to the [Simon Funk](http://sifter.org/~simon/journal/20061211.html) matrix factorization using stocastic gradient decent.   There is a good [ATT Research Paper](http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf) on this topics.  Our implmentation was to be contained in a class that match the 'API' from the morning sprint.   

The idea is that we have a ratings matrix $R$, and are trying to break it into a users matrix $U$ and a movie matrix $M$ such that:

$$R = U \ M$$

Since we do not know $U$ and $M$, we must find it by finding the error, then iteratively changing the matrixes until we have the least error possible.

$$ \mbox{squared error} =  \sum_{i,j} (r_{ij} - u_{ia} \ m_{aj})^2 $$

The SGD method is to go through all the ratings and use the following update rules:

$$ u_i \ = u_i + \ \gamma ( e_{ij} m_i - \lambda u_i ) $$

$$ m_i  \ = m_i + \ \gamma ( e_{ij} u_j - \lambda m_i ) $$

You can see our class below


    class MatrixFactorizationRecommender(object):
        
        def __init__(self,n_features=8,learning_rate = 0.001, 
                     regularization=0.02,optimizer_pct_improvement_criterion=2,
                     verbose=False):
            
            """Initializes the parameters of the model.
            """
            self.n_features = n_features
            self.learning_rate = learning_rate
            self.regularization = regularization
            self.optimizer_pct_improvement_criterion = optimizer_pct_improvement_criterion
            self.verbose = verbose
    
        def fit(self,matrix):
            """Like the scikit learn fit methods, this method 
            should take the ratings data as an input and should
            compute and store the matrix factorization. It should assign
            some class variables like n_users, which depend on the
            ratings_mat data.
    
            It can return nothing
            """
            self.matrix = matrix
            
            n_users = self.matrix.shape[0]
            n_movies = self.matrix.shape[1]
            n_already_rated = self.matrix.nonzero()[0].size
            user_mat = np.random.rand(
                n_users*self.n_features).reshape([n_users, self.n_features])
            movie_mat = np.random.rand(
                n_movies*self.n_features).reshape([self.n_features, n_movies])
    
            
            optimizer_iteration_count = 0
            sse_accum = 0
            if self.verbose:
                print("Optimizaiton Statistics")
                print("Iterations | Mean Squared Error  |  Percent Improvement")
            while (optimizer_iteration_count < 2 or (pct_improvement > self.optimizer_pct_improvement_criterion)):
                old_sse = sse_accum
                sse_accum = 0
                for i in range(n_users):
                    for j in range(n_movies):
                        if self.matrix[i, j] > 0:
                            error = self.matrix[i, j] - \
                                np.dot(user_mat[i, :], movie_mat[:, j])
                            sse_accum += error**2
                            for k in range(self.n_features):
                                user_mat[i, k] = user_mat[
                                    i, k] + self.learning_rate * (2 * error * movie_mat[k, j] - self.regularization * user_mat[i, k])
                                movie_mat[k, j] = movie_mat[
                                    k, j] + self.learning_rate * (2 * error * user_mat[i, k] - self.regularization * movie_mat[k, j])
                pct_improvement = 100 * (old_sse - sse_accum) / old_sse
                if self.verbose:
                        print("%d \t\t %f \t\t %f" % (
                            optimizer_iteration_count, sse_accum / n_already_rated, pct_improvement))
                old_sse = sse_accum
                optimizer_iteration_count += 1
                
            self.user_mat = user_mat
            self.movie_mat = movie_mat
            self.recommendations = self.user_mat.dot(self.movie_mat)
    
        def pred_one_user(self,index):
            """Returns the predicted rating for a single
            user.
            """
            return self.recommendations[index]
        
        def pred_all_users(self):
            """Returns the predicted rating for all users/items.
            """
            return self.recommendations
           
        def top_n_recs(self,index,n):
            """Takes user_id argument and number argument.
    
            Returns that number of items with the highest predicted ratings,
            after removing items that user has already rated.
            """
            scores = self.matrix[0,:].nonzero()[0]
            predictions = self.pred_one_user(index)
            predictions[scores] = -1
            return np.argsort(predictions)[::-1][:n]
    
    


Now we can look at the relative performance and predictions from this morning!


    from SVD import MatrixFactorizationRecommender
    NMF = MatrixFactorizationRecommender(n_features=20)
    t1 = time()
    NMF.fit(df.values)
    print "Time to Fit: ", time()-t1

    Time to Fit:  51.0261509418


The fit time is signficantly longer than the Item-Item recommender.  What about predictions.


    t1=time()
    print "Prediction: ", np.round(NMF.pred_one_user(0)[:10],0)
    print "Actual:     ", df.values[0,:10]
    print "Time to Predict 1 User: ", time()-t1
    t1 = time()
    predictions = NMF.pred_all_users()
    print "Time to predict 943 Users:", time()-t1

    Prediction:  [ 4.  3.  3.  3.  3.  4.  4.  4.  4.  4.]
    Actual:      [ 5.  3.  4.  3.  3.  5.  4.  1.  5.  3.]
    Time to Predict 1 User:  0.00182199478149
    Time to predict 943 Users: 0.00263500213623


So in this case the predictions is just the matrix multiplication found in the fit.  It is significantly faster to make predictions.   In fact I cheated by saving the predictions in the class.  This is really just a lookup.

It does seem the predictions are not as good as the Item-Item recommendor.   Lets look at the movie predictions.


    idf.loc[NMF.top_n_recs(0,10),:]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1632</th>
      <td>� k�ldum klaka (Cold Fever) (1994)</td>
      <td>08-Mar-1996</td>
    </tr>
    <tr>
      <th>1405</th>
      <td>When Night Is Falling (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>Madame Butterfly (1995)</td>
      <td>20-Sep-1996</td>
    </tr>
    <tr>
      <th>1616</th>
      <td>Hugo Pool (1997)</td>
      <td>01-Jan-1997</td>
    </tr>
    <tr>
      <th>1463</th>
      <td>Stars Fell on Henrietta, The (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>1648</th>
      <td>Big One, The (1997)</td>
      <td>27-Mar-1998</td>
    </tr>
    <tr>
      <th>1583</th>
      <td>Symphonie pastorale, La (1946)</td>
      <td>01-Jan-1946</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>Hedd Wyn (1992)</td>
      <td>01-Jan-1992</td>
    </tr>
    <tr>
      <th>1481</th>
      <td>Gate of Heavenly Peace, The (1995)</td>
      <td>10-May-1996</td>
    </tr>
    <tr>
      <th>1599</th>
      <td>Guantanamera (1994)</td>
      <td>16-May-1997</td>
    </tr>
  </tbody>
</table>
</div>



These are wildly different predictions than this morning.  In order to get a feel for which better we need to come up with a way to validate our strategy. 

##Validations

If this was a uppervised problem, we would make a training and testing set, cross validate on the training set, then measure the error on the test set.  We can do that here by removing rating from the matrix, and checking if which recommendor is making a better predictions on the test set.


    def validation_score(REC,frac_row, frac_col,matrix,scoring_func):
        import numpy as np
        row_test = np.random.choice(matrix.shape[0],int(matrix.shape[0]*frac_row),replace=True)
        col_test = np.random.choice(matrix.shape[1],int(matrix.shape[1]*frac_col),replace=True)
        matrix_copy = matrix.copy()
        matrix_copy[row_test,:][:,col_test] = 0.
        REC.fit(matrix_copy)
        pred = REC.pred_all_users()
        return scoring_func(matrix[row_test,:][:,col_test],pred[row_test,:][:,col_test])
    
    def mse_recommendor(truth_mat, pred_mat):
            """
            Computes mean-squared-error between a sparse and a dense matrix.  Does not include the 0's from
            the sparse matrix in computation (treats them as missing)
            """
            #get mask of non-zero, mean-square of those, divide by count of those
            nonzero_idx = truth_mat.nonzero()
            mse = (np.array(truth_mat[nonzero_idx] - pred_mat[nonzero_idx])**2).mean()
            return mse


    IIR = ItemItemRecommender(neighborhood_size=20)
    validation_score(IIR,0.3,0.3,df.values,mse_recommendor)




    0.57135223474377161




    NMF = MatrixFactorizationRecommender(n_features=20)
    validation_score(NMF,0.3,0.3,df.values,mse_recommendor)




    0.88865196231368859



Our mean error on the Item-Item is about half a rating, while the error on the Matrix-Factorization is almost a full rating.   In this case, I will probably go with the Item-Item

The NMF, on the other hand, has a number of turning parameters.  I performed a gridsearch offline, and found parameters that improved the fit.   The way Item-Item is done, this is not possible.   Below is a fit that is much better.


    NMF = MatrixFactorizationRecommender(n_features=20,learning_rate=0.01,regularization=0.01)
    validation_score(NMF,0.3,0.3,df.values,mse_recommendor)




    0.47107202771056506




    NMF = MatrixFactorizationRecommender(n_features=20,learning_rate=0.01,regularization=0.01)
    NMF.fit(df.values)
    idf.loc[NMF.top_n_recs(0,10),:]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>407</th>
      <td>Close Shave, A (1995)</td>
      <td>28-Apr-1996</td>
    </tr>
    <tr>
      <th>1367</th>
      <td>Mina Tannenbaum (1994)</td>
      <td>01-Jan-1994</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Lawrence of Arabia (1962)</td>
      <td>01-Jan-1962</td>
    </tr>
    <tr>
      <th>660</th>
      <td>High Noon (1952)</td>
      <td>01-Jan-1952</td>
    </tr>
    <tr>
      <th>646</th>
      <td>Ran (1985)</td>
      <td>01-Jan-1985</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Secrets &amp; Lies (1996)</td>
      <td>04-Oct-1996</td>
    </tr>
    <tr>
      <th>918</th>
      <td>City of Lost Children, The (1995)</td>
      <td>01-Jan-1995</td>
    </tr>
    <tr>
      <th>477</th>
      <td>Philadelphia Story, The (1940)</td>
      <td>01-Jan-1940</td>
    </tr>
    <tr>
      <th>428</th>
      <td>Day the Earth Stood Still, The (1951)</td>
      <td>01-Jan-1951</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>When We Were Kings (1996)</td>
      <td>14-Feb-1997</td>
    </tr>
  </tbody>
</table>
</div>



We can see the movies recommended by the tuned recommendor are different from before.  They are also, at a glance, much more related to the moviews this user rated.  This improves the recommendor, but the search took over an hour to run, and I have no idea how much better we can make it.

One final note is that we noticed with Matrix-Factorization based recomendors is that when the number of features are low, the highest rated and most rated movies dominate the suggestions.   You haven't seen 'Titatic'?  Watch it!  'Casablanca'?  Watch it!   It defaults to a popularity recommendor instead of a personalized recommendor.   


    
