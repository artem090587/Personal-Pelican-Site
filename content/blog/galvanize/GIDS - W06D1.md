Title: Galvanize - Week 06 - Day 1
Date: 2015-07-06 10:20
Modified: 2015-06-03 10:30
Category: Galvanize
Tags: data-science, galvanize, nlp, PCA, SVD
Slug: galvanize-data-science-06-01
Authors: Bryan Smith
Summary: Today we covered clustering PCA and SVD

#Galvanize Immersive Data Science

##Week 6 - Day 1

Today we had a quiz where we continued yesterday's work of vectorizing documents in a corpus, and finding similar documents.   The idea of two day's miniquiz was to treat a document like a corpus, and treat sentence's like documents.   We then clustered related sentences together to find related topics in the document.   

##PCA

Our morning lesson was on Dimensional Reduction with the focus on Priciple Components Analysis.  PCA is finding orthogonal directions in a feature space that are directed along the direction of maximum variation.  The idea is that you can represent the variation of the data while reducing the number of features in the data set.  Our exploration of this begins with sklearn's digit dataset.

##Digits

The digits dataset is 100 hand written images of numeric digits.   The version we are exploring are 64 pixel square images.  You can see the images below.


    import matplotlib.pyplot as plt
    import numpy as np
    %matplotlib inline
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    plt.figure(figsize=(14,14))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(X[i].reshape(8,8),cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_1_0.png)


We can see that most of the digits are clear and distinct for a human to classify.   An algorithm, may have some work cutout for it when it comes to separate some of the 4's, 7's, and 9's.  Despite being 8x8 images, it looks like a large number of them are also cut off.   This likely has to do with the compression that took them from the original 128x128 resolution.

The PCA algorithm looks for the direction of maximum variation, then projects along the data onto the axis.   The next direction is the new direction of maximimun variation.  The plot of the variance explained is called a Scree plot, and you can see this for the digits dataset.  


    scaler = StandardScaler()
    sX = scaler.fit_transform(X)
    pca = PCA(63)
    psX = pca.fit_transform(X)
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.plot(range(63),pca.explained_variance_ratio_,'r--')
    plt.ylabel('Precent Variance Explained')
    plt.xlabel('Number of Components')
    plt.subplot(122)
    plt.plot(range(63),np.cumsum(pca.explained_variance_ratio_),'g--')
    plt.ylabel('Precent Cumlative Variance Explained')
    plt.xlabel('Number of Components')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_3_0.png)


Each image is 64 features, and we are reduce the number of features, or compress the data, while maintain a fixed amount of variation.   We can keep 80% by compressiong it to 10 features, or 90% if we compress it down to 20 features.  

Lets look at what these images look like for the 1st, and the 10th compoents.


    scaler = StandardScaler()
    sX = scaler.fit_transform(X)
    pca = PCA(1)
    psX = pca.fit_transform(X)
    Xcompressed = pca.inverse_transform(psX)
    plt.figure(figsize=(14,14))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(Xcompressed[i].reshape(8,8),cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_5_0.png)


We see that the first axis projects the image into how much like a 3 or a 6 does it look like.  The 6 makes sense because it fills the most space that is not used by other numbers.  That is the bottom left quadrant.  The 3 is other side.  I was surprised the 9 was not the other digit shown.  Reguardless, the 3's look a lot like 3's, and the 6's look a lot like 6's.  The problem is that the information about the other digits are lost.  We can look at the first 10 components.


    scaler = StandardScaler()
    sX = scaler.fit_transform(X)
    pca = PCA(10)
    psX = pca.fit_transform(X)
    Xcompressed = pca.inverse_transform(psX)
    plt.figure(figsize=(14,14))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(Xcompressed[i].reshape(8,8),cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_7_0.png)


The images are much clearer than 1 component, but we can see them as being blur.  We have effectively compressed the images.   A human can easily tell which digit most of the images are, but there are some that are still clear.  The first 5 is a good example of that.  Finally, lets look at 20 compoents.


    scaler = StandardScaler()
    sX = scaler.fit_transform(X)
    pca = PCA(20)
    psX = pca.fit_transform(X)
    Xcompressed = pca.inverse_transform(psX)
    plt.figure(figsize=(14,14))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(Xcompressed[i].reshape(8,8),cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_9_0.png)


At this point all the images are clear enough for a human with fair eyesight to make out.   We have cut the features size by a 1/3 while maintaining human readibility.  This is the idea behind PCA.  We can still maintain the predictive ability or use case while reducing the feature size.

You can also see how different digits cluster in the space of the PCA compoents.  Below is a plot of the first PCA direction on the x axis, and the second PCA direction on the y axis.   The color identify which digit the data is for.  We can see that there is overlap, but the same digits cluster together, and related digits cluster next to each other.  


    scaler = StandardScaler()
    sX = scaler.fit_transform(X)
    pca = PCA(2)
    psX = pca.fit_transform(X)
    plt.figure(figsize=(14,14))
    for i in range(10):
        plt.plot(psX[y==i,0],psX[y==i,1],color=plt.cm.Set1(i*3),marker='o',alpha=0.9,label=str(i),lw=0,markersize=10)
    plt.legend()




    <matplotlib.legend.Legend at 0x10db23690>




![png](http://www.bryantravissmith.com/img/GW06D1/output_11_1.png)



    plt.figure(figsize=(14,14))
    ax = plt.gca()
    for i in range(psX.shape[0]):
        #print psX[i,0],psX[i,1],y[i]
        ax.annotate(str(y[i]),xy=(psX[i,0],psX[i,1]),color=plt.cm.Set1(y[i]*3), fontsize=14,alpha=0.6)
    plt.xlim([-40,40])
    plt.ylim([-40,40])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_12_0.png)


We replaced each point with the label of the digit.  In this plot it is clear that 5's are difficult to distingust with the first 2 compoents, but zeros and ones seem well clusted.   The 3's, 9's and 7's have a lot of overlap.   As we add more features, we saw visually the images become more distinct.

##Cars

The cars dataset is a default package in R, and we are going to use it to predict the mpg of the car using a linear fit.   The goals is to do it for the dataset, then for the PCA compoents of the dataset.  The goal is to see that as we increase the pca, our fit matches the full it, but it is well appoximated by a subset of features.  


    import pandas as pd
    cars = pd.read_table('data/cars.tsv',header=None)
    cars = pd.concat([pd.DataFrame(cars.loc[:,0].str.split().tolist()),cars[[1]]],axis=1,ignore_index=True)
    cars = cars[cars[3] != '?'].iloc[:,:8]
    cars = cars.convert_objects(convert_numeric=True)
    cars.columns = ['mpg', 'cylinders','displacement','horsepower','weight','acceleration','model_year', 'origin']
    cars.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>8</td>
      <td>307</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>8</td>
      <td>350</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>8</td>
      <td>318</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>8</td>
      <td>304</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>8</td>
      <td>302</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We can produce a scree plot to where we can estiate the number of components necessary for making consistent predictions with the full dataset.   


    sca = StandardScaler()
    pca = PCA(8)
    sCar = sca.fit_transform(cars.values)
    pCar = pca.fit_transform(sCar)
    plt.plot(range(1,9),np.cumsum(pca.explained_variance_ratio_))




    [<matplotlib.lines.Line2D at 0x109c12bd0>]




![png](http://www.bryantravissmith.com/img/GW06D1/output_16_1.png)


From this plt we see most fo the variance can be explained with 3 to 4 components depending on if we want 90% or 95% of the variation explained.   I will expect that our adjusted R-square will get better for the first 3 or 4 PCA components, then level off to the adjusted R-square of the full fit.


    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    y = cars[['mpg']].values
    X = cars.drop('mpg',axis=1).values
    x_trn,x_tst,y_trn,y_tst = train_test_split(X,y,test_size=0.2)
    
    sca = StandardScaler()
    pca = PCA(7)
    sx_trn = sca.fit_transform(x_trn)
    px_trn = pca.fit_transform(sx_trn)
    
    lin1 = LinearRegression()
    lin1.fit(x_trn,y_trn)
    y_pred = lin1.predict(x_tst)
    print "Full Fit Adjusted R-Square", 1-(1-r2_score(y_tst,y_pred))*(len(y_tst)-1)/(len(y_tst)-x_trn.shape[1]-1)
    
    for i in range(1,8):
        pca = PCA(i)
        sx_trn = sca.fit_transform(x_trn)
        px_trn = pca.fit_transform(sx_trn)
        linP = LinearRegression()
        linP.fit(px_trn,y_trn)
        
        sx_tst = sca.transform(x_tst)
        px_tst = pca.transform(sx_tst)
        y_pred = linP.predict(px_tst)
        adj_r2 = 1-(1-r2_score(y_tst,y_pred))*(len(y_tst)-1)/(len(y_tst)-i-1)
        print "PCA Adjusted R-Square {}".format(i), adj_r2

    Full Fit Adjusted R-Square 0.817772756314
    PCA Adjusted R-Square 1 0.69397778532
    PCA Adjusted R-Square 2 0.696494130123
    PCA Adjusted R-Square 3 0.791628786462
    PCA Adjusted R-Square 4 0.797026468934
    PCA Adjusted R-Square 5 0.807152321105
    PCA Adjusted R-Square 6 0.813375746265
    PCA Adjusted R-Square 7 0.817772756314


The above first are just a section of data.  Because we have a small dataset and a 20% split on the data to run on the test set, we will see large variation in the results.  Reguardless of the quality of the fit on the test set, what we always see is the we have dramatic increases in the adjusted r-square until we get to the 4th component, then we get incremental increase. 

##Singular Value Decomposition

The idea behind SVD's is that any matrix can be factors into the product of 3 matrixes.  We have a dataset of book reviews for users, and we want to try to decompose these relationships into the idea of topics.  The users will have relationships to topics, and books will have relationships to topics.  This is the beginnings of recommender systems that we will be exploring this week.




    df  = pd.read_csv('data/book_reviews.csv')
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>User-ID</th>
      <th>ISBN</th>
      <th>Book-Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1083</td>
      <td>277195</td>
      <td>0060391626</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1084</td>
      <td>277195</td>
      <td>0060502258</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1089</td>
      <td>277195</td>
      <td>0060987561</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1098</td>
      <td>277195</td>
      <td>0316666343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1099</td>
      <td>277195</td>
      <td>0316734837</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    df.drop('Unnamed: 0',axis=1,inplace=True)
    df = df.set_index(['User-ID','ISBN']).unstack().fillna(-1)
    df.columns = df.columns.droplevel()
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>ISBN</th>
      <th>0006493580</th>
      <th>000649840X</th>
      <th>0006512135</th>
      <th>0006513204</th>
      <th>0006514855</th>
      <th>0006547834</th>
      <th>0006550576</th>
      <th>0006550681</th>
      <th>0006550789</th>
      <th>0007110928</th>
      <th>...</th>
      <th>8495618605</th>
      <th>8497593588</th>
      <th>8804342838</th>
      <th>8806142100</th>
      <th>8806143042</th>
      <th>8807813025</th>
      <th>8817106100</th>
      <th>8845205118</th>
      <th>8873122933</th>
      <th>8885989403</th>
    </tr>
    <tr>
      <th>User-ID</th>
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
      <th>243</th>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>254</th>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>507</th>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>638</th>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>805</th>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 6092 columns</p>
</div>




    U,S,V = np.linalg.svd(df.values)
    S




    array([  3.73197977e+03,   3.28163606e+02,   2.34177766e+02, ...,
             2.95911201e+00,   2.63616054e+00,   2.25714589e+00])



Now that we have decomposed the User x Book review matrix, we can look at the singular values S.  The rule of thumb is that the energy, $S^2$, is the analogous to the explained variance of the PCA.  We can make a plot of this verse the number of components we want to inclunde in the reduction, and estimate how much information loss will be experience.


    plt.plot(range(1,len(S)+1),np.cumsum(S**2/np.sum(S**2)),'r--')
    plt.xlabel('Number of SVD Components Included')
    plt.ylabel('Percent of Total Energy Included')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_25_0.png)


We see that we can maintain 90% of the energy with 1 first 500 components of the 2500 possible singular values.   We can keep 95% with 1000 components.  Interestingly we include 72% of the energy with only 1 component.  What we are going is try to get a feel for the concepts/topics produced by the SVD algorithm.  We are going to load in some book meta data, and try to find characteristic titles for each topics.


    meta = pd.read_csv('data/book_meta.csv',sep=";",usecols=['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher'])
    meta.head()

    /Library/Python/2.7/site-packages/pandas/io/parsers.py:1164: DtypeWarning: Columns (3) have mixed types. Specify dtype option on import or set low_memory=False.
      data = self._reader.read(nrows)





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0195153448</td>
      <td>Classical Mythology</td>
      <td>Mark P. O. Morford</td>
      <td>2002</td>
      <td>Oxford University Press</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002005018</td>
      <td>Clara Callan</td>
      <td>Richard Bruce Wright</td>
      <td>2001</td>
      <td>HarperFlamingo Canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0060973129</td>
      <td>Decision in Normandy</td>
      <td>Carlo D'Este</td>
      <td>1991</td>
      <td>HarperPerennial</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0374157065</td>
      <td>Flu: The Story of the Great Influenza Pandemic...</td>
      <td>Gina Bari Kolata</td>
      <td>1999</td>
      <td>Farrar Straus Giroux</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0393045218</td>
      <td>The Mummies of Urumchi</td>
      <td>E. J. W. Barber</td>
      <td>1999</td>
      <td>W. W. Norton &amp;amp; Company</td>
    </tr>
  </tbody>
</table>
</div>



The basic idea is our U matrix has rows that represent users and columns that represent latent topcs.  The $\Sigma$ (S) matrix presents the weight of each topic, and the V matrix has rows that represent topics and columns that represent books.   

We are going to go through the V matrix and find the topic books for the first few topics to see if we can learn what the latent association is.


    top20_book_indexes = np.argsort(V[0,0:])[::-1][:20]
    top20_book_isbns = df.columns[top20_book_indexes]
    meta[meta.ISBN.isin(top20_book_isbns)]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>1841721522</td>
      <td>New Vegetarian: Bold and Beautiful Recipes for...</td>
      <td>Celia Brooks Brown</td>
      <td>2001</td>
      <td>Ryland Peters &amp;amp; Small Ltd</td>
    </tr>
    <tr>
      <th>1229</th>
      <td>3257061269</td>
      <td>Der Alchimist.</td>
      <td>Paulo Coelho</td>
      <td>2003</td>
      <td>Diogenes Verlag, Z�?¼rich</td>
    </tr>
    <tr>
      <th>2587</th>
      <td>3423105518</td>
      <td>Name Der Rose</td>
      <td>Umberto Eco</td>
      <td>0</td>
      <td>Distribooks Int'l+inc</td>
    </tr>
    <tr>
      <th>3028</th>
      <td>1844262553</td>
      <td>Free</td>
      <td>Paul Vincent</td>
      <td>2003</td>
      <td>Upfront Publishing</td>
    </tr>
    <tr>
      <th>3217</th>
      <td>3548603203</td>
      <td>Artemis Fowl.</td>
      <td>Eoin Colfer</td>
      <td>2003</td>
      <td>Ullstein TB-Vlg</td>
    </tr>
    <tr>
      <th>5042</th>
      <td>3257229364</td>
      <td>Endstation Venedig. Commissario Brunettis zwei...</td>
      <td>Donna Leon</td>
      <td>1996</td>
      <td>Diogenes Verlag</td>
    </tr>
    <tr>
      <th>7909</th>
      <td>8817106100</td>
      <td>Oceano Mare</td>
      <td>Alessandro Baricco</td>
      <td>0</td>
      <td>Biblioteca Universale Rizzoli</td>
    </tr>
    <tr>
      <th>8839</th>
      <td>3423202327</td>
      <td>M�?¶rder ohne Gesicht.</td>
      <td>Henning Mankell</td>
      <td>1999</td>
      <td>Dtv</td>
    </tr>
    <tr>
      <th>10786</th>
      <td>3423201509</td>
      <td>Die Weiss Lowin / Contemporary German Lit</td>
      <td>Henning Mankell</td>
      <td>2002</td>
      <td>Distribooks</td>
    </tr>
    <tr>
      <th>11766</th>
      <td>8807813025</td>
      <td>Novocento, Un Monologo</td>
      <td>Alessandro Baricco</td>
      <td>2003</td>
      <td>Distribooks Inc</td>
    </tr>
    <tr>
      <th>12842</th>
      <td>3379015180</td>
      <td>Schlafes Bruder</td>
      <td>Robert Schneider</td>
      <td>1994</td>
      <td>Reclam, Leipzig</td>
    </tr>
    <tr>
      <th>27672</th>
      <td>3462032283</td>
      <td>Zw�?¶lf.</td>
      <td>Nick McDonell</td>
      <td>2003</td>
      <td>Kiepenheuer &amp;amp; Witsch</td>
    </tr>
    <tr>
      <th>37478</th>
      <td>3492238696</td>
      <td>Balzac und die kleine chinesische Schneiderin.</td>
      <td>Dai Sijie</td>
      <td>2003</td>
      <td>Piper</td>
    </tr>
    <tr>
      <th>38240</th>
      <td>3462028189</td>
      <td>Crazy</td>
      <td>Benjamin Lebert</td>
      <td>2000</td>
      <td>Kiepenheuer &amp;amp; Witsch GmbH &amp;amp; Co. KG, Ve...</td>
    </tr>
    <tr>
      <th>51792</th>
      <td>3250600555</td>
      <td>Monsieur Ibrahim und die Blumen des Koran. Erz...</td>
      <td>Eric-Emmanuel Schmitt</td>
      <td>2002</td>
      <td>Ammann</td>
    </tr>
    <tr>
      <th>91342</th>
      <td>3442414199</td>
      <td>Generation X. Geschichten f�?¼r eine immer sch...</td>
      <td>Douglas Coupland</td>
      <td>1994</td>
      <td>Goldmann</td>
    </tr>
  </tbody>
</table>
</div>



The first topic looks like it has a strong relationship to german titles.  They were all published in the late 90's or early 00's.   I am currious how many reviews these books received.


    print np.sum(df.values[:,top20_book_indexes]>=0,axis=0)
    print np.round(np.average(df.values[:,top20_book_indexes],weights=df.values[:,top20_book_indexes]>=0,axis=0),1)

    [2 1 1 1 2 5 1 1 1 1 1 2 2 2 2 2 3 1 1 4]
    [ 5.   9.   9.   9.   4.5  2.   8.   8.   8.   8.   8.   4.   4.   4.   4.
      4.   2.7  7.   7.   2. ]


These books have been reviewed between 1 to 5 times, and the average rating is between 2 and 9.  German authors seems to be the biggest association between these books.

Lets repeat this for a few more topics



    top20_book_indexes = np.argsort(V[1,0:])[::-1][:20]
    top20_book_isbns = df.columns[top20_book_indexes]
    meta[meta.ISBN.isin(top20_book_isbns)]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107</th>
      <td>0786868716</td>
      <td>The Five People You Meet in Heaven</td>
      <td>Mitch Albom</td>
      <td>2003</td>
      <td>Hyperion</td>
    </tr>
    <tr>
      <th>408</th>
      <td>0316666343</td>
      <td>The Lovely Bones: A Novel</td>
      <td>Alice Sebold</td>
      <td>2002</td>
      <td>Little, Brown</td>
    </tr>
    <tr>
      <th>522</th>
      <td>0312195516</td>
      <td>The Red Tent (Bestselling Backlist)</td>
      <td>Anita Diamant</td>
      <td>1998</td>
      <td>Picador USA</td>
    </tr>
    <tr>
      <th>706</th>
      <td>0446672211</td>
      <td>Where the Heart Is (Oprah's Book Club (Paperba...</td>
      <td>Billie Letts</td>
      <td>1998</td>
      <td>Warner Books</td>
    </tr>
    <tr>
      <th>748</th>
      <td>0385504209</td>
      <td>The Da Vinci Code</td>
      <td>Dan Brown</td>
      <td>2003</td>
      <td>Doubleday</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>0345361792</td>
      <td>A Prayer for Owen Meany</td>
      <td>John Irving</td>
      <td>1990</td>
      <td>Ballantine Books</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>0743418174</td>
      <td>Good in Bed</td>
      <td>Jennifer Weiner</td>
      <td>2002</td>
      <td>Washington Square Press</td>
    </tr>
    <tr>
      <th>1863</th>
      <td>0446610038</td>
      <td>1st to Die: A Novel</td>
      <td>James Patterson</td>
      <td>2002</td>
      <td>Warner Vision</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>067976402X</td>
      <td>Snow Falling on Cedars</td>
      <td>David Guterson</td>
      <td>1995</td>
      <td>Vintage Books USA</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>0316569321</td>
      <td>White Oleander : A Novel</td>
      <td>Janet Fitch</td>
      <td>1999</td>
      <td>Little, Brown</td>
    </tr>
    <tr>
      <th>2143</th>
      <td>059035342X</td>
      <td>Harry Potter and the Sorcerer's Stone (Harry P...</td>
      <td>J. K. Rowling</td>
      <td>1999</td>
      <td>Arthur A. Levine Books</td>
    </tr>
    <tr>
      <th>2290</th>
      <td>0385484518</td>
      <td>Tuesdays with Morrie: An Old Man, a Young Man,...</td>
      <td>MITCH ALBOM</td>
      <td>1997</td>
      <td>Doubleday</td>
    </tr>
    <tr>
      <th>2910</th>
      <td>0380718340</td>
      <td>Cruel &amp;amp; Unusual (Kay Scarpetta Mysteries (...</td>
      <td>Patricia D. Cornwell</td>
      <td>1994</td>
      <td>Avon</td>
    </tr>
    <tr>
      <th>3939</th>
      <td>0316096199</td>
      <td>Lucky : A Memoir</td>
      <td>Alice Sebold</td>
      <td>2002</td>
      <td>Back Bay Books</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>0375727345</td>
      <td>House of Sand and Fog</td>
      <td>Andre Dubus III</td>
      <td>2000</td>
      <td>Vintage Books</td>
    </tr>
    <tr>
      <th>5070</th>
      <td>014028009X</td>
      <td>Bridget Jones's Diary</td>
      <td>Helen Fielding</td>
      <td>1999</td>
      <td>Penguin Books</td>
    </tr>
    <tr>
      <th>5873</th>
      <td>0312966091</td>
      <td>Three To Get Deadly : A Stephanie Plum Novel (...</td>
      <td>Janet Evanovich</td>
      <td>1998</td>
      <td>St. Martin's Paperbacks</td>
    </tr>
    <tr>
      <th>5887</th>
      <td>0671001795</td>
      <td>Two for the Dough</td>
      <td>Janet Evanovich</td>
      <td>1996</td>
      <td>Pocket</td>
    </tr>
    <tr>
      <th>6401</th>
      <td>0609804138</td>
      <td>The Sweet Potato Queens' Book of Love</td>
      <td>JILL CONNER BROWNE</td>
      <td>1999</td>
      <td>Three Rivers Press</td>
    </tr>
    <tr>
      <th>7852</th>
      <td>0553280341</td>
      <td>B Is for Burglar (Kinsey Millhone Mysteries (P...</td>
      <td>Sue Grafton</td>
      <td>1986</td>
      <td>Bantam</td>
    </tr>
  </tbody>
</table>
</div>



These topics seem to be with popular fiction, maybe for teenagers.  One more topics before I move one.


    top20_book_indexes = np.argsort(V[2,0:])[::-1][:20]
    top20_book_isbns = df.columns[top20_book_indexes]
    meta[meta.ISBN.isin(top20_book_isbns)]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>042518630X</td>
      <td>Purity in Death</td>
      <td>J.D. Robb</td>
      <td>2002</td>
      <td>Berkley Publishing Group</td>
    </tr>
    <tr>
      <th>368</th>
      <td>0515128554</td>
      <td>Heart of the Sea (Irish Trilogy)</td>
      <td>Nora Roberts</td>
      <td>2000</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>0373484224</td>
      <td>Stanislaski Brothers (Silhouette Promo)</td>
      <td>Nora Roberts</td>
      <td>2000</td>
      <td>Silhouette</td>
    </tr>
    <tr>
      <th>2784</th>
      <td>051513287X</td>
      <td>Face the Fire (Three Sisters Island Trilogy)</td>
      <td>Nora Roberts</td>
      <td>2002</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>3163</th>
      <td>0515114693</td>
      <td>Born in Fire</td>
      <td>Nora Roberts</td>
      <td>1994</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>4544</th>
      <td>0515132020</td>
      <td>Heaven and Earth (Three Sisters Island Trilogy)</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>4546</th>
      <td>0515131229</td>
      <td>Dance upon the Air (Three Sisters Island Trilogy)</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>8977</th>
      <td>039914840X</td>
      <td>Three Fates</td>
      <td>Nora Roberts</td>
      <td>2002</td>
      <td>Putnam Publishing Group</td>
    </tr>
    <tr>
      <th>10929</th>
      <td>0515128546</td>
      <td>Tears of the Moon (Irish Trilogy)</td>
      <td>Nora Roberts</td>
      <td>2000</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>11633</th>
      <td>0399148248</td>
      <td>Midnight Bayou</td>
      <td>Nora Roberts</td>
      <td>2001</td>
      <td>Putnam Publishing Group</td>
    </tr>
    <tr>
      <th>15358</th>
      <td>0399149848</td>
      <td>Birthright</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Putnam Publishing Group</td>
    </tr>
    <tr>
      <th>15513</th>
      <td>0553265741</td>
      <td>Sacred Sins</td>
      <td>Nora Roberts</td>
      <td>1990</td>
      <td>Bantam Books</td>
    </tr>
    <tr>
      <th>16198</th>
      <td>0515136379</td>
      <td>Key of Knowledge (Key Trilogy (Paperback))</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>16199</th>
      <td>0515136530</td>
      <td>Key of Valor (Roberts, Nora. Key Trilogy, 3.)</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Jove Pubns</td>
    </tr>
    <tr>
      <th>16203</th>
      <td>051513628X</td>
      <td>Key of Light (Key Trilogy (Paperback))</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>17665</th>
      <td>0373483503</td>
      <td>Macgregor Brides (Macgregors)</td>
      <td>Nora Roberts</td>
      <td>1997</td>
      <td>Silhouette</td>
    </tr>
    <tr>
      <th>18414</th>
      <td>0425183971</td>
      <td>Reunion in Death</td>
      <td>J. D. Robb</td>
      <td>2002</td>
      <td>Berkley Publishing Group</td>
    </tr>
    <tr>
      <th>18476</th>
      <td>0425189031</td>
      <td>Portrait in Death</td>
      <td>Nora Roberts</td>
      <td>2003</td>
      <td>Berkley Publishing Group</td>
    </tr>
    <tr>
      <th>19926</th>
      <td>0515126772</td>
      <td>Jewels of the Sun (Irish Trilogy)</td>
      <td>Nora Roberts</td>
      <td>2004</td>
      <td>Jove Books</td>
    </tr>
    <tr>
      <th>26087</th>
      <td>0515116750</td>
      <td>Born in Ice</td>
      <td>Nora Roberts</td>
      <td>1996</td>
      <td>Berkley Publishing Group</td>
    </tr>
  </tbody>
</table>
</div>



We have an obvious connection here for topic three:  Nora Roberts.   The singular value decomposition of these book and user ratings is interesting, but I am not seeing how it is scalable.

We can also investigate the user topic preferences by exploring the U matrix.  Remember that each row is a user, and each column a topic.  Lets looka the first user.


    top10 = np.argsort(U[0,:])[::-1][:10]
    print top10
    print np.round((U[0,top10]-np.mean(U[0,:]))/np.std(U[0,:]),2)

    [2108 2038 2037 2272 2128 2015 1820 2025 2054 2232]
    [ 4.38  4.19  3.62  3.48  3.45  3.36  3.3   3.25  3.    2.98]


The first user is associated with these topics, and has standard scores associated with them that at or above 3.   
We can look at this user's book reviews.  


    original = pd.read_csv('data/book_reviews.csv')
    print original[original['User-ID']==df.index[0]].shape
    user1_fav = original[(original['User-ID']==df.index[0]) & (original['Book-Rating'] >= 9)]
    user1_fav

    (66, 4)





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>User-ID</th>
      <th>ISBN</th>
      <th>Book-Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1147</th>
      <td>9955</td>
      <td>243</td>
      <td>0060915544</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1152</th>
      <td>9962</td>
      <td>243</td>
      <td>0316601950</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1156</th>
      <td>9966</td>
      <td>243</td>
      <td>0316776963</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>9976</td>
      <td>243</td>
      <td>0375400117</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>9994</td>
      <td>243</td>
      <td>0425163407</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>10005</td>
      <td>243</td>
      <td>0446364800</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



The first user rated 66 books, and 6 of the books were rated at a 9 or a 10.  These are this users favorite books.  We will want to see if these books are associated with topics that user has been grouped into.  


    user1_fav_indexes = np.where(df.columns.isin(user1_fav.ISBN.values))[0]
    np.argmax(V[:,user1_fav_indexes],axis=0)




    array([ 155,   77,  112,  291,   35, 2893])



They are not characteristic of the topics that the first user is associated with.  Lets how deep we need to go before we find the topics overlap between this person's best reviewed books and the topics this person is associated with.


    np.where(np.argsort(V[:,user1_fav_indexes[0]],axis=0)[::-1]==top10[0])




    (array([532]),)



This is very interesting for when we get to recommendation systems.  This user's top books are not remotely associated with the top topics the user is associated with.   This user did rate 66 books, and so the average topic could be different than the individual books would be associated with.   But if we are going to recommend a new book to this user, would we do it by topic, or by book similarity, or by user similarity.   To be continued...

##Senate

We were given data for Senate voting records and asked to visualize the polarization for the 101st through 111th congresses.  The hard part of this project was cleaning and formating the data.   I will save you that processes because it was not informative.  

Once the data was clean, we use distance measurements between the voting records, mapped these differences onto a 2D manifold, then displayed them with a color coding of 'Republican' and 'Democrate'.


    from sklearn.manifold import MDS
    from scipy.spatial.distance import pdist,squareform
    mds = MDS()
    
    plt.figure(figsize=(14,10))
    for i in range(101,112):
        df1 = pd.read_csv("data/senate/s{}.csv".format(i))
        df2 = df1.fillna(0).replace([1,2,3],1).replace([4,5,6],-1).replace([7,8,9,0],0)
        d101 = squareform(pdist(df2.iloc[:,10:].values)) 
        mds.fit(d101)
        mask = df2.party.values==100
        plt.subplot(3,4,i-100)
        plt.plot(mds.embedding_[mask,0],mds.embedding_[mask,1],'ro')
        plt.plot(mds.embedding_[~mask,0],mds.embedding_[~mask,1],'bo')
        plt.xlim([-150,150])
        plt.ylim([-150,150])
        
        plt.title("Senate {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_45_0.png)


In this space, the voting vectors of republicans and democrats are very different.  As the years have move one, we see that it looks like they are drifiting appart.   We can remake this graph using PCA instead distance mapped onto a 2D manifold.  This coordinate system will capture directions of most variation.


    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    plt.figure(figsize=(14,10))
    for i in range(101,112):
        df1 = pd.read_csv("data/senate/s{}.csv".format(i))
        df2 = df1.fillna(0).replace([1,2,3],1).replace([4,5,6],-1).replace([7,8,9,0],0.000001)
        X = pca.fit_transform(df2.iloc[:,10:].values)
        
        mask = df2.party.values==100
        plt.subplot(3,4,i-100)
        plt.plot(X[mask,0],X[mask,1],'ro')
        plt.plot(X[~mask,0],X[~mask,1],'bo')
        plt.title("Senate {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D1/output_47_0.png)


This representation does not illustrate the same level of polarization as the previous graphs, but it does have a more sensible interpretation.   
