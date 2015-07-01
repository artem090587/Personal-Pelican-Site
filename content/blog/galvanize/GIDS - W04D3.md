Title: Galvanize - Week 04 - Day 3
Date: 2015-06-24 10:20
Modified: 2015-06-24 10:30
Category: Galvanize
Tags: data-science, galvanize, SVM, support vector machines
Slug: galvanize-data-science-04-03
Authors: Bryan Smith
Summary: Today we covered SVMs.

#Galvanize Immersive Data Science

##Week 4 - Day 3

Our quiz toda was about making change.  Given a sufficient amount US coins what is the minimum number of coins needed to give change for a specificied amount?   

We were suppose to build a function to do this.  My solution was to sort the list of coins from largest to smallest.   We then make the maximum amount of change using the largest denomination, then continue this until we have given change back.


    
    def find_change(coins,value):
        coins.sort(reverse=True)
        n = 0
        v = value
        for x in coins:
            m =  v / x
            n += m
            v = v - m*x       
        return n
    
    coins = [1,5,10,25]
    print "Correct Answer 4, Your Answer: ", find_change(coins,100)
    print "Correct Answer 8, Your Answer: ",find_change(coins,74) 

    Correct Answer 4, Your Answer:  4
    Correct Answer 8, Your Answer:  8


After the quiz we had a lecture on Support Vector Machines.  The afternoon lecture was on kernel tricks for SVMs.   

##Morning: Maximal Margin Classifier

We learned that a support vectore machine is a maximum margin classifier, trying to construct a hyper plane that maximize the margin of linearly seperable data.   

To help get a feel between SVMs and other classifiers we looked up some made up data about the number of hours emailing and number of hours spent at the gym for a people labeled as a data scientist or not a data scientist.

This dataset is special in that it is linearly seperable and easily displayed in two dimensions


    %matplotlib inline
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('data/data_scientist.csv')
    plt.figure(figsize=(16,8))
    ax = plt.subplot(1,1,1)
    df[df.data_scientist==0].plot(kind='scatter',x='email_hours',y='gym_hours',color='steelblue',s=100,ax=ax, label="Not Data Scientist")
    df[df.data_scientist==1].plot(kind='scatter',x='email_hours',y='gym_hours',color='seagreen',s=100, ax=ax, label='Data Scientist')
    plt.legend(loc=2)
    plt.xlabel('Hours Emailing')
    plt.ylabel('Hours At Gym')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/(output_3_0.png)


## Margin of Logistic Regression Boundary

We learned previously that logistic regression minimizes the log-loss function:

$$ - \ \sum_{i=1}^m [ \ y_i \ log(h_\theta (x_i)) \ + \ (1-y_i) \ log(1-h_\theta (x_i)) \ ]$$

Where $$h_\theta(x) = \frac{1}{1+e^{x\theta}}$$.  

This is different that explicitly maximizing the margin of a decision boundary.  

We can fit a logistic regression model on our data that has 100% accuracy.  I have plotted the theta dotted the data on the x-axis, and the data scientist status on the y-axis.  


    from sklearn.linear_model import LogisticRegression
    lin = LogisticRegression(fit_intercept=True,C=10)
    lin.fit(df[['email_hours','gym_hours']].values,df.data_scientist.values)
    z = df[['email_hours','gym_hours']].values.dot(lin.coef_.T)+lin.intercept_
    df['z'] = z
    plt.figure(figsize=(16,8))
    ax = plt.subplot(1,1,1)
    df[df.data_scientist==0].plot(kind='scatter',x='z',y='data_scientist',color='steelblue',label='Not Data Scientist', s=100,ax=ax)
    df[df.data_scientist==1].plot(kind='scatter',x='z',y='data_scientist',color='seagreen',label='Data Scientist', s=100, ax=ax)
    zp = np.linspace(-12,6,100)
    yp = 1/(1+np.exp(-zp))
    plt.plot(zp,yp,color='indianred',lw=4,alpha=0.5,label='Logistic Fit')
    plt.legend(loc=2)
    plt.show()
    
    print "Accuracy: ", lin.score(df[['email_hours','gym_hours']].values,df.data_scientist.values)


![png](http://www.bryantravissmith.com/img/GW04D3/output_5_0.png)


    Accuracy:  1.0


###Fun Fact!

I origianlly fit this without regulariaiton, but the algorithm was converging before finding the optimal solution.  I added a regularization term to make a better (100%) fit to the classification.  

2. Write a function to compute and plot the decision boundary. Remember `y` is `0` at the decision boundary when
   the probability of a positive class is `0.5`. You should also define a range over one of your features (`gym_hours`
   for example) and compute the `email_hours` at the decision boundary.
   
  


    x = np.linspace(0,50,100)
    y = np.linspace(0,50,100)
    xx, yy = np.meshgrid(x,y)
    Z = lin.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = Z.reshape(xx.shape)
    #zz = 1/(1+np.exp(xx*lin.coef_[0,0]+yy*lin.coef_[0,1]+lin.intercept_)) 
    extent = [0,50,0,50]
    plt.figure(figsize=(14,8))
    plt.pcolormesh(xx, yy, zz, cmap='winter',alpha=0.1, ) #plt.cm.Paired)
    
    ax = plt.subplot(1,1,1)
    #b = (0.5 - lin.intercept_)/lin.coef_[0,1]
    #m = -lin.coef_[0,0]/lin.coef_[0,1]
    #plt.plot(x,m*x+b,linestyle='--',color='black')
    df[df.data_scientist==0].plot(kind='scatter',x='email_hours',y='gym_hours',color='steelblue',s=100,label='Not Data Scientist',alpha=0.8,ax=ax)
    df[df.data_scientist==1].plot(kind='scatter',x='email_hours',y='gym_hours',color='seagreen',s=100,label='Data Scientist',alpha=0.8, ax=ax)
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.xlabel('Hours Emailing')
    plt.ylabel('Hours At Gym')
    plt.legend(loc=2)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_8_0.png)




The distance each point is from the margin is given by the following equations 

   $$\mbox{distance} = \frac{\beta_0+\beta x^T}{||\beta||}$$
   
A line perpendicular to a line will always have a slope that is $\frac{1}{\mbox{slope}}.  We can use this equation and make our plot to illustrate the distance from the margin using the size property.  



    def distance(x,slopes,intercept):
        return (intercept+x.dot(slopes))/np.sqrt(intercept**2+slopes.T.dot(slopes))
    
    df['s'] = np.abs(250*distance(df[['email_hours','gym_hours']].values,lin.coef_.T,lin.intercept_))
    x = np.linspace(0,50,100)
    y = np.linspace(0,50,100)
    xx, yy = np.meshgrid(x,y)
    Z = lin.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = Z.reshape(xx.shape)
    #zz = 1/(1+np.exp(xx*lin.coef_[0,0]+yy*lin.coef_[0,1]+lin.intercept_)) 
    extent = [0,50,0,50]
    plt.figure(figsize=(14,8))
    plt.pcolormesh(xx, yy, zz, cmap='winter',alpha=0.1, ) #plt.cm.Paired)
    
    ax = plt.subplot(1,1,1)
    #b = (0.5 - lin.intercept_)/lin.coef_[0,1]
    #m = -lin.coef_[0,0]/lin.coef_[0,1]
    #plt.plot(x,m*x+b,linestyle='--',color='black')
    df[df.data_scientist==0].plot(kind='scatter',x='email_hours',
                                  y='gym_hours',color='steelblue',
                                  s=df[df.data_scientist==0].s,
                                  label='Not Data Scientist',
                                  alpha=0.8,ax=ax)
    df[df.data_scientist==1].plot(kind='scatter',x='email_hours',
                                  y='gym_hours',color='seagreen',
                                  s=df[df.data_scientist==1].s,
                                  label='Data Scientist',alpha=0.8, 
                                  ax=ax)
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.xlabel('Hours Emailing')
    plt.ylabel('Hours At Gym')
    plt.legend(loc=2)
    plt.show()



![png](http://www.bryantravissmith.com/img/GW04D3/output_11_0.png)


##Margin of Support Vector Machines

We learned that the SVM is a maximal margin classifier which in theory would have a larger margin than Logistic Regression. We will go through the same process that we just did for logistic regression.   


    from sklearn.svm import SVC
    
    svc = SVC(kernel='linear')
    svc.fit(df[['email_hours','gym_hours']].values,df.data_scientist.values)
    svc.score(df[['email_hours','gym_hours']].values,df.data_scientist.values)




    1.0




    svc.coef_,svc.intercept_
    x = np.linspace(0,50,100)
    y = np.linspace(0,50,100)
    xx, yy = np.meshgrid(x,y)
    zz = distance(np.c_[xx.ravel(),yy.ravel()],svc.coef_.T,svc.intercept_)
    zz = zz.reshape(xx.shape)
    
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = Z.reshape(xx.shape)
    
    df['ss'] = np.abs(100*distance(df[['email_hours','gym_hours']].values,svc.coef_.T,svc.intercept_))
    
    plt.figure(figsize=(14,8))
    plt.pcolormesh(xx, yy, zz, cmap='winter',alpha=0.1, ) #plt.cm.Paired)
    
    
    b = (-svc.intercept_)/svc.coef_[0,1]
    m = -svc.coef_[0,0]/svc.coef_[0,1]
    plt.plot(x,m*x+b,linestyle='--',color='black')
    plt.plot(x,m*x+b+svc.intercept_,linestyle='-',color='black')
    plt.plot(x,m*x+b-svc.intercept_,linestyle='-',color='black')
    ax = plt.subplot(1,1,1)
    df[df.data_scientist==0].plot(kind='scatter',x='email_hours',
                                  y='gym_hours',color='steelblue',
                                  s=df[df.data_scientist==0].ss,
                                  label='Not Data Scientist',
                                  alpha=0.8,ax=ax)
    df[df.data_scientist==1].plot(kind='scatter',x='email_hours',
                                  y='gym_hours',color='seagreen',
                                  s=df[df.data_scientist==1].ss,
                                  label='Data Scientist',alpha=0.8, 
                                  ax=ax)
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.xlabel('Hours Emailing')
    plt.ylabel('Hours At Gym')
    plt.legend(loc=2)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_14_0.png)


The SVM classifier seems to match my intuition for what the optimal boundary should be, and the logistic regression did not capture that.  If we had a new data point at 10 hours of emailing and 15 hours at the gym, I would expect that person to be a data scientist because it is closer to the cluster of data scientists.  The logistic regression fit we did previously would have classifed it as a non-data scientist.  This is despit it is so far from the cluster.

Just because this is my intuition does not mean that its correct.   There are problems that this intuition is incorrect.   That is why you might have some insight into the problem because picking a classifier.  That is also why you cross validate and test on a unseen test set.   

##Scaling

We just worked a problem where the scaling of the two variables are the same, but if they are not this will mess with the results the svm will produce.   The distance measurements change with units.  It is standard practice to scale variables before fitting an SVM on a data set.   If we do not, we can get different results.  


    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv('data/non_sep.csv',header=None)
    x = df.loc[1:,1:2].values
    y = df.loc[1:,3].values
    pipeline = Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear'))])
    pipeline.fit(x, y)
    svc = pipeline.named_steps['svc']
    
    xvals = np.linspace(-5,5,100)
    yvals = (svc.intercept_+svc.coef_[0,0]*xvals)/(-1*svc.coef_[0,1])
    mask = y==1
    
    xx,yy = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
    Z = pipeline.predict(np.c_[xx.ravel(),yy.ravel()])
    zz = Z.reshape(xx.shape)
    
    plt.figure(figsize=(14,8))
    plt.pcolormesh(xx,yy,zz,cmap='BrBG',alpha=0.2)
    plt.plot(x[mask,0],x[mask,1],color='seagreen',marker='o', markersize=10,lw=0,label='1\'s')
    plt.plot(x[~mask,0],x[~mask,1],color='burlywood',marker='o',markersize=10,lw=0,label='0\'s')
    plt.legend()
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_16_0.png)


In this case the data is not linearly seperable.  You will have noticed I change the colors from the previous spot to values that contrast better.  That way you can see when values are misclassified more easily.  The overlap is not very much in this instance, so we can still git fair accuracy.   A 5-Fold cross validation shows above 90% accuracy.


    from sklearn.cross_validation import cross_val_score
    cross_val_score(pipeline,x,y,scoring='accuracy',cv=5).mean()




    0.93000000000000005



##Tuning SVMs

The SVM has a C parameter that acts like $\frac{1}{\lambda}$ for regularization in Lasso and Ridge Regression.  Changing this values allows allowing error.  We get get a feel for how it affect the accuracy of a prediction by scanning through a number of values of this tuning paramter.


    cv_score = []
    Cs = np.logspace(-3, 1, 100)
    for c in Cs:
        pipeline = Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear', C=c))])
        cv_score.append(cross_val_score(pipeline, x, y, scoring='accuracy', cv=10).mean())
    
    plt.figure(figsize=(14,8))
    plt.plot(Cs,cv_score,lw=2,color='seagreen')
    plt.xlabel("Tuning Paramter")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.ylim([0.9,.95])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_20_0.png)


We can see that by changing the tuning parameter that we change the accuracy of the SVM, but there are similar accuracies for different C values.  To show what the algorithm is doing we can plot the decision boundaries for two very different C's on the non-seperable dataset we have started investigating.  


    pipeline1 = Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear',C=.01))])
    pipeline1.fit(x, y)
    pipeline2= Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear',C=10))])
    pipeline2.fit(x, y)
    
    mask = y==1
    xx,yy = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
    Z1 = pipeline1.predict(np.c_[xx.ravel(),yy.ravel()])
    zz1 = Z1.reshape(xx.shape)
    
    Z2 = pipeline2.predict(np.c_[xx.ravel(),yy.ravel()])
    zz2 = Z2.reshape(xx.shape)
    
    plt.figure(figsize=(14,8))
    plt.pcolormesh(xx,yy,zz1,cmap='BrBG',alpha=0.2)
    plt.pcolormesh(xx,yy,zz2,cmap='binary',alpha=0.2)
    plt.plot(x[mask,0],x[mask,1],color='seagreen',marker='o', markersize=10,lw=0,label='1\'s')
    plt.plot(x[~mask,0],x[~mask,1],color='burlywood',marker='o',markersize=10,lw=0,label='0\'s')
    plt.legend()
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_22_0.png)


We see that the two regions have different slopes.  I mapped a binary (black/white) colormap over the large C fit.   This makes the background darker.  The smaller C has the hyperplane seperator with a more negative slope than the larger C.   By changing the hyperparameter, we are ultimately move the hyperplane that is attempting to fit the data.

##Kernels

SVM's can accept a kernal argument that effectively maps the data into a higher dimension, potentially making the data linearably seperable when it otherwise might not be.  We can illustrate this with a simple example using random data.  We will have some data that is seperable, but not linearably seperable.   


    x1r = 2*np.random.random((200,1))-1
    x2r = 2*np.random.random((200,1))-1
    
    x3r = x1r*x1r+x2r*x2r
    
    yr = np.sqrt(x1r*x1r+x2r*x2r) < .5
    
    mask = yr==0
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(x1r[mask],x2r[mask],'ro')
    plt.plot(x1r[~mask],x2r[~mask],'bo')
    plt.subplot(1,2,2)
    plt.plot(x1r[mask],x3r[mask],'ro')
    plt.plot(x1r[~mask],x3r[~mask],'bo')
    plt.axhline(y=0.25)
    plt.show()



![png](http://www.bryantravissmith.com/img/GW04D3/output_24_0.png)


By transforming the data into a third dimention we take a seperable data and make in seperable by a hyperplane.  How we can train an SVM on this data, and find the decision boundary.  Two common kernals to fit data two is a gaussian kernal and a polynomial kernal.  This allow for making non-linear decision surfaces.  Lets first look at the RBF kernel.

##RBF Kernel


    def plot_surface(kernel,C=1,degree=3,gamma=1,*args):
        pipeline = Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel=kernel,C=C,degree=degree,gamma=1,*args))])
        pipeline.fit(x,y)
        svc_rbf = pipeline.named_steps['svc']
    
    
        mask = y==1
        xx,yy = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
        Z = pipeline.predict(np.c_[xx.ravel(),yy.ravel()])
        zz = Z.reshape(xx.shape)
    
    
        plt.figure(figsize=(14,8))
        plt.pcolormesh(xx,yy,zz,cmap='BrBG',alpha=0.3)
        plt.plot(x[mask,0],x[mask,1],color='seagreen',marker='o', markersize=10,lw=0,label='1\'s')
        plt.plot(x[~mask,0],x[~mask,1],color='burlywood',marker='o',markersize=10,lw=0,label='0\'s')
        plt.legend()
        plt.xlim([-5,5])
        plt.ylim([-5,5])
        plt.title("SVM with " + kernel + " Kernel, C = "+str(C))
        plt.show()
        
    plot_surface('rbf')


![png](http://www.bryantravissmith.com/img/GW04D3/output_26_0.png)


The decision surface of this SVM is clearly not linear in the data, but it is linear in a hyperspace the data is projected into.   We can look at how the accuracy changes as we turn the model.  As well as how the deciion surfaces changes.


    cv_score = []
    Cs = np.logspace(-3, 1, 100)
    for c in Cs:
        pipeline = Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel='rbf', C=c))])
        cv_score.append(cross_val_score(pipeline, x, y, scoring='accuracy', cv=10).mean())
    
    plt.figure(figsize=(14,8))
    plt.plot(Cs,cv_score,lw=2,color='seagreen')
    plt.xlabel("Tuning Paramter")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.ylim([0.89,.95])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_28_0.png)



    plot_surface('rbf',C=100)


![png](http://www.bryantravissmith.com/img/GW04D3/output_29_0.png)



    plot_surface('rbf',C=1000)


![png](http://www.bryantravissmith.com/img/GW04D3/output_30_0.png)


We can see that as we increase the turning parameter, which is lower the regulization, we get very curvy decision surfaces.

##Polynomial

The polynomial kernal allows for some curvature to the decision surface, but usally less than that fit by the RBF kernel.   We can see a degree polynomial curve below.


    plot_surface('poly')


![png](http://www.bryantravissmith.com/img/GW04D3/output_32_0.png)


We can look at the surface as we change the turning parameter and the degree of the kernel.


    plot_surface('rbf',C=100,degree=3)


![png](http://www.bryantravissmith.com/img/GW04D3/output_34_0.png)



    plot_surface('rbf',C=1,degree=5)


![png](http://www.bryantravissmith.com/img/GW04D3/output_35_0.png)


We see that the hypersurface changes its form adn has more curvature as we increase the size of these parameters.  

##Grid Search

Ideally we will want to find the best model to fit the data we are given.  This is difficult to do by hand, so we can use a grid search strategy to find the best model on our training data.


    from sklearn.grid_search import GridSearchCV
    parameters = {'svc__degree':np.arange(2,10),'svc__C':np.logspace(-2,2,100)}
    pipeline = Pipeline([('scaler', StandardScaler()),
                        ('svc', SVC(kernel='poly'))])
    clf = GridSearchCV(pipeline, parameters,scoring='accuracy')
    clf.fit(x,y)
    print "Best Accuracy:", clf.best_score_
    print "Best Parameters:", clf.best_params_

    Best Accuracy: 0.94
    Best Parameters: {'svc__degree': 3, 'svc__C': 0.058570208180566671}



    plot_surface('poly',C=clf.best_params_['svc__C'],degree=clf.best_params_['svc__degree'])


![png](http://www.bryantravissmith.com/img/GW04D3/output_38_0.png)


##Multi-Classification

We can also use SVM's, and other classifiers, with datesets that have multiple classifications.   An example is the digits dataset that has hand written digits as images and we are attempting to classify them.

The two methods is 1 vs all, which makes a classifier for each classification.  Then each data point is scored by each classifier.  The highest scored predictor is then classified as a positive example in that classifier.
In 1 vs 1, a classifier is made for each pair of data, and then there is a vote.  The classification with the most votes win.   We will be using both of these with a SVM on the digit data.

Because we are going to be using linear kernals, there is a optimize algorithm in sklearn that we will be using.


    from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
    from sklearn.datasets import load_digits
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    data = load_digits(n_class=10)
    images = data.images.reshape(1797,64)
    
    LinVsAll = OneVsRestClassifier(LinearSVC())
    LinVsOne = OneVsOneClassifier(LinearSVC())
    
    plt.figure(figsize=(14,8))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(data.images[i],cmap='Greys')
    
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D3/output_40_0.png)


    


The data are 8x8 pixle images of numbers that are hand written.  There are 10 classifications available in tis dataset.  We will train both multiclassification methods on the dataset and check the test accuracy.


    x_train, x_test, y_train, y_test = train_test_split(images,data.target)
    
    LinVsAll.fit(x_train,y_train)
    y_pred = LinVsAll.predict(x_test)
    
    print "One Vs All"
    print "Accuracy: ", accuracy_score(y_test,y_pred)
    print "Recall: ", recall_score(y_test,y_pred,average='weighted')
    print "Precision: ", precision_score(y_test,y_pred,average='weighted')
    
    
    print ""
    print "One Vs One"
    LinVsOne.fit(x_train,y_train)
    y_pred = LinVsOne.predict(x_test)
    
    print "Accuracy: ", accuracy_score(y_test,y_pred)
    print "Recall: ", recall_score(y_test,y_pred,average='weighted')
    print "Precision: ", precision_score(y_test,y_pred,average='weighted')

    One Vs All
    Accuracy:  0.942222222222
    Recall:  0.942222222222
    Precision:  0.942982349293
    
    One Vs One
    Accuracy:  0.975555555556
    Recall:  0.975555555556
    Precision:  0.976581128748


In this case we have the One vs One method being more accurate.  I suspect that we find some values of 7,9, and 4 that are pretty similar, and the voting helps seperate them while the best score does not. 

##Real World Data

We have some biological data that we are asked to predict if a given sample comes from stool or tissue.  This problem is interesting to me because the data is not structured in a way for us to answer it.  We have to restructure the data set!


    df_b = pd.read_csv('data/bio.csv').drop('Group',axis=1)
    df_b.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Taxon</th>
      <th>Patient</th>
      <th>Tissue</th>
      <th>Stool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Firmicutes</td>
      <td>1</td>
      <td>136</td>
      <td>4182</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Firmicutes</td>
      <td>2</td>
      <td>1174</td>
      <td>703</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Firmicutes</td>
      <td>3</td>
      <td>408</td>
      <td>3946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Firmicutes</td>
      <td>4</td>
      <td>831</td>
      <td>8605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Firmicutes</td>
      <td>5</td>
      <td>693</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




    df_b = df_b.pivot('Patient','Taxon')
    df_b




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">Tissue</th>
      <th colspan="5" halign="left">Stool</th>
    </tr>
    <tr>
      <th>Taxon</th>
      <th>Actinobacteria</th>
      <th>Bacteroidetes</th>
      <th>Firmicutes</th>
      <th>Other</th>
      <th>Proteobacteria</th>
      <th>Actinobacteria</th>
      <th>Bacteroidetes</th>
      <th>Firmicutes</th>
      <th>Other</th>
      <th>Proteobacteria</th>
    </tr>
    <tr>
      <th>Patient</th>
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
      <td>1590</td>
      <td>67</td>
      <td>136</td>
      <td>195</td>
      <td>2469</td>
      <td>4</td>
      <td>0</td>
      <td>4182</td>
      <td>18</td>
      <td>1821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>0</td>
      <td>1174</td>
      <td>42</td>
      <td>839</td>
      <td>2</td>
      <td>0</td>
      <td>703</td>
      <td>2</td>
      <td>661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>259</td>
      <td>85</td>
      <td>408</td>
      <td>316</td>
      <td>4414</td>
      <td>300</td>
      <td>5</td>
      <td>3946</td>
      <td>43</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>568</td>
      <td>143</td>
      <td>831</td>
      <td>202</td>
      <td>12044</td>
      <td>7</td>
      <td>7</td>
      <td>8605</td>
      <td>40</td>
      <td>83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1102</td>
      <td>678</td>
      <td>693</td>
      <td>116</td>
      <td>2310</td>
      <td>9</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>678</td>
      <td>4829</td>
      <td>718</td>
      <td>527</td>
      <td>3053</td>
      <td>377</td>
      <td>209</td>
      <td>717</td>
      <td>12</td>
      <td>547</td>
    </tr>
    <tr>
      <th>7</th>
      <td>260</td>
      <td>74</td>
      <td>173</td>
      <td>357</td>
      <td>395</td>
      <td>58</td>
      <td>651</td>
      <td>33</td>
      <td>11</td>
      <td>2174</td>
    </tr>
    <tr>
      <th>8</th>
      <td>424</td>
      <td>169</td>
      <td>228</td>
      <td>106</td>
      <td>2651</td>
      <td>233</td>
      <td>254</td>
      <td>80</td>
      <td>11</td>
      <td>767</td>
    </tr>
    <tr>
      <th>9</th>
      <td>548</td>
      <td>106</td>
      <td>162</td>
      <td>67</td>
      <td>1195</td>
      <td>21</td>
      <td>10</td>
      <td>3196</td>
      <td>14</td>
      <td>76</td>
    </tr>
    <tr>
      <th>10</th>
      <td>201</td>
      <td>73</td>
      <td>372</td>
      <td>203</td>
      <td>6857</td>
      <td>83</td>
      <td>381</td>
      <td>32</td>
      <td>6</td>
      <td>795</td>
    </tr>
    <tr>
      <th>11</th>
      <td>42</td>
      <td>30</td>
      <td>4255</td>
      <td>392</td>
      <td>483</td>
      <td>75</td>
      <td>359</td>
      <td>4361</td>
      <td>6</td>
      <td>666</td>
    </tr>
    <tr>
      <th>12</th>
      <td>109</td>
      <td>51</td>
      <td>107</td>
      <td>28</td>
      <td>2950</td>
      <td>59</td>
      <td>51</td>
      <td>1667</td>
      <td>25</td>
      <td>3994</td>
    </tr>
    <tr>
      <th>13</th>
      <td>51</td>
      <td>2473</td>
      <td>96</td>
      <td>12</td>
      <td>1541</td>
      <td>183</td>
      <td>2314</td>
      <td>223</td>
      <td>22</td>
      <td>816</td>
    </tr>
    <tr>
      <th>14</th>
      <td>310</td>
      <td>102</td>
      <td>281</td>
      <td>305</td>
      <td>1307</td>
      <td>204</td>
      <td>33</td>
      <td>2377</td>
      <td>32</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




    df_b = df_b.stack(level=0)
    df_b




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Taxon</th>
      <th>Actinobacteria</th>
      <th>Bacteroidetes</th>
      <th>Firmicutes</th>
      <th>Other</th>
      <th>Proteobacteria</th>
    </tr>
    <tr>
      <th>Patient</th>
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
      <th rowspan="2" valign="top">1</th>
      <th>Tissue</th>
      <td>1590</td>
      <td>67</td>
      <td>136</td>
      <td>195</td>
      <td>2469</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>4</td>
      <td>0</td>
      <td>4182</td>
      <td>18</td>
      <td>1821</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>Tissue</th>
      <td>25</td>
      <td>0</td>
      <td>1174</td>
      <td>42</td>
      <td>839</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>2</td>
      <td>0</td>
      <td>703</td>
      <td>2</td>
      <td>661</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Tissue</th>
      <td>259</td>
      <td>85</td>
      <td>408</td>
      <td>316</td>
      <td>4414</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>300</td>
      <td>5</td>
      <td>3946</td>
      <td>43</td>
      <td>18</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>Tissue</th>
      <td>568</td>
      <td>143</td>
      <td>831</td>
      <td>202</td>
      <td>12044</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>7</td>
      <td>7</td>
      <td>8605</td>
      <td>40</td>
      <td>83</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">5</th>
      <th>Tissue</th>
      <td>1102</td>
      <td>678</td>
      <td>693</td>
      <td>116</td>
      <td>2310</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>9</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">6</th>
      <th>Tissue</th>
      <td>678</td>
      <td>4829</td>
      <td>718</td>
      <td>527</td>
      <td>3053</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>377</td>
      <td>209</td>
      <td>717</td>
      <td>12</td>
      <td>547</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">7</th>
      <th>Tissue</th>
      <td>260</td>
      <td>74</td>
      <td>173</td>
      <td>357</td>
      <td>395</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>58</td>
      <td>651</td>
      <td>33</td>
      <td>11</td>
      <td>2174</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">8</th>
      <th>Tissue</th>
      <td>424</td>
      <td>169</td>
      <td>228</td>
      <td>106</td>
      <td>2651</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>233</td>
      <td>254</td>
      <td>80</td>
      <td>11</td>
      <td>767</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">9</th>
      <th>Tissue</th>
      <td>548</td>
      <td>106</td>
      <td>162</td>
      <td>67</td>
      <td>1195</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>21</td>
      <td>10</td>
      <td>3196</td>
      <td>14</td>
      <td>76</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">10</th>
      <th>Tissue</th>
      <td>201</td>
      <td>73</td>
      <td>372</td>
      <td>203</td>
      <td>6857</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>83</td>
      <td>381</td>
      <td>32</td>
      <td>6</td>
      <td>795</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">11</th>
      <th>Tissue</th>
      <td>42</td>
      <td>30</td>
      <td>4255</td>
      <td>392</td>
      <td>483</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>75</td>
      <td>359</td>
      <td>4361</td>
      <td>6</td>
      <td>666</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">12</th>
      <th>Tissue</th>
      <td>109</td>
      <td>51</td>
      <td>107</td>
      <td>28</td>
      <td>2950</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>59</td>
      <td>51</td>
      <td>1667</td>
      <td>25</td>
      <td>3994</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">13</th>
      <th>Tissue</th>
      <td>51</td>
      <td>2473</td>
      <td>96</td>
      <td>12</td>
      <td>1541</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>183</td>
      <td>2314</td>
      <td>223</td>
      <td>22</td>
      <td>816</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">14</th>
      <th>Tissue</th>
      <td>310</td>
      <td>102</td>
      <td>281</td>
      <td>305</td>
      <td>1307</td>
    </tr>
    <tr>
      <th>Stool</th>
      <td>204</td>
      <td>33</td>
      <td>2377</td>
      <td>32</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




    df_b = df_b.reset_index()
    df_b




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Taxon</th>
      <th>Patient</th>
      <th>level_1</th>
      <th>Actinobacteria</th>
      <th>Bacteroidetes</th>
      <th>Firmicutes</th>
      <th>Other</th>
      <th>Proteobacteria</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Tissue</td>
      <td>1590</td>
      <td>67</td>
      <td>136</td>
      <td>195</td>
      <td>2469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Stool</td>
      <td>4</td>
      <td>0</td>
      <td>4182</td>
      <td>18</td>
      <td>1821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Tissue</td>
      <td>25</td>
      <td>0</td>
      <td>1174</td>
      <td>42</td>
      <td>839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Stool</td>
      <td>2</td>
      <td>0</td>
      <td>703</td>
      <td>2</td>
      <td>661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Tissue</td>
      <td>259</td>
      <td>85</td>
      <td>408</td>
      <td>316</td>
      <td>4414</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>Stool</td>
      <td>300</td>
      <td>5</td>
      <td>3946</td>
      <td>43</td>
      <td>18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>Tissue</td>
      <td>568</td>
      <td>143</td>
      <td>831</td>
      <td>202</td>
      <td>12044</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>Stool</td>
      <td>7</td>
      <td>7</td>
      <td>8605</td>
      <td>40</td>
      <td>83</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>Tissue</td>
      <td>1102</td>
      <td>678</td>
      <td>693</td>
      <td>116</td>
      <td>2310</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>Stool</td>
      <td>9</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>Tissue</td>
      <td>678</td>
      <td>4829</td>
      <td>718</td>
      <td>527</td>
      <td>3053</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
      <td>Stool</td>
      <td>377</td>
      <td>209</td>
      <td>717</td>
      <td>12</td>
      <td>547</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
      <td>Tissue</td>
      <td>260</td>
      <td>74</td>
      <td>173</td>
      <td>357</td>
      <td>395</td>
    </tr>
    <tr>
      <th>13</th>
      <td>7</td>
      <td>Stool</td>
      <td>58</td>
      <td>651</td>
      <td>33</td>
      <td>11</td>
      <td>2174</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
      <td>Tissue</td>
      <td>424</td>
      <td>169</td>
      <td>228</td>
      <td>106</td>
      <td>2651</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8</td>
      <td>Stool</td>
      <td>233</td>
      <td>254</td>
      <td>80</td>
      <td>11</td>
      <td>767</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9</td>
      <td>Tissue</td>
      <td>548</td>
      <td>106</td>
      <td>162</td>
      <td>67</td>
      <td>1195</td>
    </tr>
    <tr>
      <th>17</th>
      <td>9</td>
      <td>Stool</td>
      <td>21</td>
      <td>10</td>
      <td>3196</td>
      <td>14</td>
      <td>76</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>Tissue</td>
      <td>201</td>
      <td>73</td>
      <td>372</td>
      <td>203</td>
      <td>6857</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10</td>
      <td>Stool</td>
      <td>83</td>
      <td>381</td>
      <td>32</td>
      <td>6</td>
      <td>795</td>
    </tr>
    <tr>
      <th>20</th>
      <td>11</td>
      <td>Tissue</td>
      <td>42</td>
      <td>30</td>
      <td>4255</td>
      <td>392</td>
      <td>483</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>Stool</td>
      <td>75</td>
      <td>359</td>
      <td>4361</td>
      <td>6</td>
      <td>666</td>
    </tr>
    <tr>
      <th>22</th>
      <td>12</td>
      <td>Tissue</td>
      <td>109</td>
      <td>51</td>
      <td>107</td>
      <td>28</td>
      <td>2950</td>
    </tr>
    <tr>
      <th>23</th>
      <td>12</td>
      <td>Stool</td>
      <td>59</td>
      <td>51</td>
      <td>1667</td>
      <td>25</td>
      <td>3994</td>
    </tr>
    <tr>
      <th>24</th>
      <td>13</td>
      <td>Tissue</td>
      <td>51</td>
      <td>2473</td>
      <td>96</td>
      <td>12</td>
      <td>1541</td>
    </tr>
    <tr>
      <th>25</th>
      <td>13</td>
      <td>Stool</td>
      <td>183</td>
      <td>2314</td>
      <td>223</td>
      <td>22</td>
      <td>816</td>
    </tr>
    <tr>
      <th>26</th>
      <td>14</td>
      <td>Tissue</td>
      <td>310</td>
      <td>102</td>
      <td>281</td>
      <td>305</td>
      <td>1307</td>
    </tr>
    <tr>
      <th>27</th>
      <td>14</td>
      <td>Stool</td>
      <td>204</td>
      <td>33</td>
      <td>2377</td>
      <td>32</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




    df_b['Location'] = np.where(df_b.level_1 == 'Tissue',1,0)
    df_b




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Taxon</th>
      <th>Patient</th>
      <th>level_1</th>
      <th>Actinobacteria</th>
      <th>Bacteroidetes</th>
      <th>Firmicutes</th>
      <th>Other</th>
      <th>Proteobacteria</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Tissue</td>
      <td>1590</td>
      <td>67</td>
      <td>136</td>
      <td>195</td>
      <td>2469</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Stool</td>
      <td>4</td>
      <td>0</td>
      <td>4182</td>
      <td>18</td>
      <td>1821</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Tissue</td>
      <td>25</td>
      <td>0</td>
      <td>1174</td>
      <td>42</td>
      <td>839</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Stool</td>
      <td>2</td>
      <td>0</td>
      <td>703</td>
      <td>2</td>
      <td>661</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Tissue</td>
      <td>259</td>
      <td>85</td>
      <td>408</td>
      <td>316</td>
      <td>4414</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>Stool</td>
      <td>300</td>
      <td>5</td>
      <td>3946</td>
      <td>43</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>Tissue</td>
      <td>568</td>
      <td>143</td>
      <td>831</td>
      <td>202</td>
      <td>12044</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>Stool</td>
      <td>7</td>
      <td>7</td>
      <td>8605</td>
      <td>40</td>
      <td>83</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>Tissue</td>
      <td>1102</td>
      <td>678</td>
      <td>693</td>
      <td>116</td>
      <td>2310</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>Stool</td>
      <td>9</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>Tissue</td>
      <td>678</td>
      <td>4829</td>
      <td>718</td>
      <td>527</td>
      <td>3053</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
      <td>Stool</td>
      <td>377</td>
      <td>209</td>
      <td>717</td>
      <td>12</td>
      <td>547</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
      <td>Tissue</td>
      <td>260</td>
      <td>74</td>
      <td>173</td>
      <td>357</td>
      <td>395</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>7</td>
      <td>Stool</td>
      <td>58</td>
      <td>651</td>
      <td>33</td>
      <td>11</td>
      <td>2174</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
      <td>Tissue</td>
      <td>424</td>
      <td>169</td>
      <td>228</td>
      <td>106</td>
      <td>2651</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8</td>
      <td>Stool</td>
      <td>233</td>
      <td>254</td>
      <td>80</td>
      <td>11</td>
      <td>767</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9</td>
      <td>Tissue</td>
      <td>548</td>
      <td>106</td>
      <td>162</td>
      <td>67</td>
      <td>1195</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>9</td>
      <td>Stool</td>
      <td>21</td>
      <td>10</td>
      <td>3196</td>
      <td>14</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>Tissue</td>
      <td>201</td>
      <td>73</td>
      <td>372</td>
      <td>203</td>
      <td>6857</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10</td>
      <td>Stool</td>
      <td>83</td>
      <td>381</td>
      <td>32</td>
      <td>6</td>
      <td>795</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>11</td>
      <td>Tissue</td>
      <td>42</td>
      <td>30</td>
      <td>4255</td>
      <td>392</td>
      <td>483</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>Stool</td>
      <td>75</td>
      <td>359</td>
      <td>4361</td>
      <td>6</td>
      <td>666</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>12</td>
      <td>Tissue</td>
      <td>109</td>
      <td>51</td>
      <td>107</td>
      <td>28</td>
      <td>2950</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>12</td>
      <td>Stool</td>
      <td>59</td>
      <td>51</td>
      <td>1667</td>
      <td>25</td>
      <td>3994</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>13</td>
      <td>Tissue</td>
      <td>51</td>
      <td>2473</td>
      <td>96</td>
      <td>12</td>
      <td>1541</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>13</td>
      <td>Stool</td>
      <td>183</td>
      <td>2314</td>
      <td>223</td>
      <td>22</td>
      <td>816</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>14</td>
      <td>Tissue</td>
      <td>310</td>
      <td>102</td>
      <td>281</td>
      <td>305</td>
      <td>1307</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>14</td>
      <td>Stool</td>
      <td>204</td>
      <td>33</td>
      <td>2377</td>
      <td>32</td>
      <td>53</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    df_b = df_b.drop(['level_1', 'Patient'], axis=1)
    df_b




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Taxon</th>
      <th>Actinobacteria</th>
      <th>Bacteroidetes</th>
      <th>Firmicutes</th>
      <th>Other</th>
      <th>Proteobacteria</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1590</td>
      <td>67</td>
      <td>136</td>
      <td>195</td>
      <td>2469</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0</td>
      <td>4182</td>
      <td>18</td>
      <td>1821</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>0</td>
      <td>1174</td>
      <td>42</td>
      <td>839</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>703</td>
      <td>2</td>
      <td>661</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>259</td>
      <td>85</td>
      <td>408</td>
      <td>316</td>
      <td>4414</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>300</td>
      <td>5</td>
      <td>3946</td>
      <td>43</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>568</td>
      <td>143</td>
      <td>831</td>
      <td>202</td>
      <td>12044</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
      <td>8605</td>
      <td>40</td>
      <td>83</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1102</td>
      <td>678</td>
      <td>693</td>
      <td>116</td>
      <td>2310</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>2</td>
      <td>50</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>678</td>
      <td>4829</td>
      <td>718</td>
      <td>527</td>
      <td>3053</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>377</td>
      <td>209</td>
      <td>717</td>
      <td>12</td>
      <td>547</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>260</td>
      <td>74</td>
      <td>173</td>
      <td>357</td>
      <td>395</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>58</td>
      <td>651</td>
      <td>33</td>
      <td>11</td>
      <td>2174</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>424</td>
      <td>169</td>
      <td>228</td>
      <td>106</td>
      <td>2651</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>233</td>
      <td>254</td>
      <td>80</td>
      <td>11</td>
      <td>767</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>548</td>
      <td>106</td>
      <td>162</td>
      <td>67</td>
      <td>1195</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>21</td>
      <td>10</td>
      <td>3196</td>
      <td>14</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>201</td>
      <td>73</td>
      <td>372</td>
      <td>203</td>
      <td>6857</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>83</td>
      <td>381</td>
      <td>32</td>
      <td>6</td>
      <td>795</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>42</td>
      <td>30</td>
      <td>4255</td>
      <td>392</td>
      <td>483</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>75</td>
      <td>359</td>
      <td>4361</td>
      <td>6</td>
      <td>666</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>109</td>
      <td>51</td>
      <td>107</td>
      <td>28</td>
      <td>2950</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>59</td>
      <td>51</td>
      <td>1667</td>
      <td>25</td>
      <td>3994</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>51</td>
      <td>2473</td>
      <td>96</td>
      <td>12</td>
      <td>1541</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>183</td>
      <td>2314</td>
      <td>223</td>
      <td>22</td>
      <td>816</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>310</td>
      <td>102</td>
      <td>281</td>
      <td>305</td>
      <td>1307</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>204</td>
      <td>33</td>
      <td>2377</td>
      <td>32</td>
      <td>53</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



And now we have our data!!!!  We can split into a train test set and see how it performs.


    y = df_b['Location'].values
    x = df_b.drop('Location', axis=1).values
    x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=.2)
    svc_lin = LinearSVC()
    parameters = {'C':np.logspace(-3,3,1000)}
    clf = GridSearchCV(svc_lin, parameters, scoring='accuracy', cv=10)
    clf.fit(x_train,y_train)
    print "Best Accuracy:", clf.best_score_
    print "Test Accuracy:", accuracy_score(y_test,clf.best_estimator_.predict(x_test))

    Best Accuracy: 0.863636363636
    Test Accuracy: 0.166666666667


Our linear SVC did not generalize to the test set at all.  Lets try a different model!


    svc_lin = SVC(kernel='rbf')
    parameters = {'C':np.logspace(-3,3,100),'gamma':np.logspace(-8,2,11)}
    clf = GridSearchCV(svc_lin, parameters, scoring='accuracy', cv=10)
    clf.fit(x_train,y_train)
    print "Best Accuracy:", clf.best_score_
    print "Test Accuracy:", accuracy_score(y_test,clf.best_estimator_.predict(x_test))

    Best Accuracy: 0.818181818182
    Test Accuracy: 0.666666666667


The RBF kernal generalized much better to the test set.  We are doing better then guessing, but I am wondering if a logistic regressor will do better.


    lin = LogisticRegression()
    parameters = {'C':np.logspace(-3,3,1000)}
    clf = GridSearchCV(lin, parameters, scoring='accuracy', cv=10)
    clf.fit(x_train,y_train)
    print "Best Accuracy:", clf.best_score_
    print "Test Accuracy:", accuracy_score(y_test,clf.best_estimator_.predict(x_test))

    Best Accuracy: 0.818181818182
    Test Accuracy: 0.333333333333


This is not the best model by far.   Before we come to a close, we will see if the SVC does better at predict than a random forest.


    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    parameters = {'n_estimators':[20,50,100,200]}
    clf = GridSearchCV(rf, parameters, scoring='accuracy', cv=10)
    clf.fit(x_train,y_train)
    print "Best Accuracy:", clf.best_score_
    print "Test Accuracy:", accuracy_score(y_test,clf.best_estimator_.predict(x_test))

    Best Accuracy: 0.818181818182
    Test Accuracy: 0.833333333333


And as expected, the random forest goes for the best generalization.   Of course, that what ensemble methods are suppose to do.  We're suppose to learn about them in depth tomorrow.  Until then....
