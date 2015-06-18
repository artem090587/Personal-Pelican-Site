Title: Galvanize - Week 03 - Day 3
Date: 2015-06-17 10:20
Modified: 2015-06-17 10:30
Category: Galvanize
Tags: data-science, galvanize, Lasso Regresion, Ridge Regression, Regularization
Slug: galvanize-data-science-03-03
Authors: Bryan Smith
Summary: The thirteenth day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered lasso and ridge regress.

#Galvanize Immersive Data Science

##Week 3 - Day 3

Today we had a miniquiz on cleaning data.  Identify missing values, incorrect format, and outliers.   It was a straight forward assignment.   I was struck with there is no clear end to data cleaning without a purpose.   It would be nice ot set a stoping criteria or variable specification so there are clear evaluation criteria.  Obviously with raw data the data has to be explored, and these criteria does not exist.   

##Morning: One-fold Cross Validation

This morning we started using cross validation on the boston data set from sklearn.  The goal is not to make the best model, but just work through th eprocess of using cross validation. 
   
Descriptions for each column in `features`:

   ```
   Attribute Information (in order):
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's
   ```
   
  


    %matplotlib inline
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import scipy.stats as sc
    from pandas.tools.plotting import scatter_matrix
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import KFold
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_val_score
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    boston = load_boston()
    features = boston.data
    target = boston.target # housing price

    


We are using the train, test split to make a training set of 70% of the data and a test set of 30% of the data.  


    train_feature, test_feature, train_target, test_target = train_test_split(features, target, test_size=0.3)

We are going to use sklearn's LinearRegresion on this data set.  There are likely issues with multicolinearity, but we will skip that for now.   We will train the data on the training set, and test it on the testing set.  


    linear = LinearRegression()
    linear.fit(train_feature, train_target)
    # You can call predict to get the predicted values for training and test
    train_predicted = linear.predict(train_feature)
    test_predicted = linear.predict(test_feature)

I wrote a mean square function to compare to sklearn's function.   I expect the test set to have larger value then the training set because the model was not trained on the test set.  


    def MSE(x1,x2):
        return np.sum(np.power(x2-x1,2))/len(x1)
    print "Train MSE: ", MSE(train_predicted,train_target),mean_squared_error(train_predicted,train_target)
    print "Test MSE: ", MSE(test_predicted,test_target), mean_squared_error(test_predicted,test_target)


    Train MSE:  21.1395296142 21.1395296142
    Test MSE:  24.6555052851 24.6555052851


##K-fold Cross Validation

In K-fold cross validation the data is split into **k** groups. One group
out of the k groups will be the test set, the rest (**k-1**) groups will
be the training set. In the next iteration, another group will be the test set,
and the rest will be the training set. The process repeats for k iterations (k-fold).
In each fold, a metric for accuracy (MSE in this case) will be calculated and
an overall average of that metric will be calculated over k-folds.

1. To do this we need to manage randomly sampling **k** folds.

2. Properly combining those **k** folds into a test and training set on
   your **on the training dataset**. Outside of the k-fold, there should be
   another set which will be referred to as the **hold-out set**.

3. Train your model on your constructed training set and evaluate on the given test set

3. Repeat steps __2__ and __3__ _k_ times.

4. Average your results of your error metric.

5. Compare the MSE for your test set in Part 1. and your K-fold cross validated error in `4.`.


    indexes = np.arange(len(train_feature))
    indexes2 = indexes.copy()
    linear = LinearRegression()
    count = len(indexes)
    mse = []
    k = 10.
    for i in range(int(k)):
        choices = np.random.choice(indexes2,int(count/k),replace=False)
        indexes2 = np.setdiff1d(indexes2,choices)
        train_index = np.setdiff1d(indexes,choices)
        linear.fit(train_feature[train_index], train_target[train_index])
        test_predicted = linear.predict(train_feature[choices])
        mse.append(mean_squared_error(train_target[choices],test_predicted))
    
    print "Avg MSE: ",sum(mse)/k

    Avg MSE:  23.9048376671



    def scorer(model,X,y):
        return mean_squared_error(y,model.predict(X))
    
    cross_val_score(LinearRegression(),train_feature,train_target,scoring=scorer,cv=10).mean()




    24.72849552718905



I made my own k-fold validation and compared it to the sklearn method.  There is an obviously efficiency in code.  I did my own wrapper for the scoring because the string method sometimes returned negative values.   

We can look at the comparison between traiing and cross validaton as we increase the sample size.  


    sample_sizes = np.arange(10,350,10)
    train_mse = []
    cross_mse = []
    for size in sample_sizes:
        linear = LinearRegression()
        
        linear.fit(train_feature[:size],train_target[:size])
        train_predicted = linear.predict(train_feature[:size])
        train_mse.append(mean_squared_error(train_predicted[:size],train_target[:size]))
        cross_mse.append(cross_val_score(linear,train_feature[:size],train_target[:size],scoring=scorer,cv=5).mean())
        
    print len(sample_sizes),len(train_mse)
    plt.figure(figsize=(10,8))
    plt.plot(sample_sizes,train_mse,color='seagreen',lw=2,alpha=0.8,label='Train Set MSE')
    plt.plot(sample_sizes,cross_mse,color='steelblue',lw=2,alpha=0.8,label='CV Set MSE')
    plt.legend()
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.ylim([0,100])
    plt.show()

    34 34



![png](http://www.bryantravissmith.com/img/output_13_1.png)


In this case we see that the two values start to converge, but the CV is higher.  This make sense because it is hold out values to test on.   It should be a better approximation to applying it to a test set. 

We can look at the difference between the test and training sets as we increase the training size.


    sample_sizes = np.arange(10,350,10)
    test_mse = []
    train_mse = []
    for size in sample_sizes:
        linear = LinearRegression()
        linear.fit(train_feature[:size],train_target[:size])
        train_predicted = linear.predict(train_feature)
        test_predicted = linear.predict(test_feature)
        train_mse.append(mean_squared_error(train_predicted,train_target))
        test_mse.append(mean_squared_error(test_predicted,test_target))
        
    plt.figure(figsize=(10,8))
    plt.plot(sample_sizes,train_mse,color='seagreen',lw=2,alpha=0.8,label='Train Set MSE')
    plt.plot(sample_sizes,test_mse,color='steelblue',lw=2,alpha=0.8,label='Test Set MSE')
    plt.legend()
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.ylim([0,100])
    plt.show()
        


![png](http://www.bryantravissmith.com/img/output_15_0.png)



    
    test_mse = []
    train_mse = []
    num_feat = range(1,len(train_feature[0,:]))
    for i in num_feat:
        linear = LinearRegression()
        linear.fit(train_feature[:,:i],train_target)
        train_predicted = linear.predict(train_feature[:,:i])
        test_predicted = linear.predict(test_feature[:,:i])
        train_mse.append(mean_squared_error(train_predicted,train_target))
        test_mse.append(mean_squared_error(test_predicted,test_target))
        
    plt.figure(figsize=(10,8))
    plt.plot(num_feat,train_mse,color='seagreen',lw=2,alpha=0.8,label='Train Set MSE')
    plt.plot(num_feat,test_mse,color='steelblue',lw=2,alpha=0.8,label='Test Set MSE')
    plt.legend()
    plt.xlabel("Number Features")
    plt.ylabel("MSE")
    plt.ylim([0,100])
    plt.show()


![png](http://www.bryantravissmith.com/img/output_16_0.png)


As we increase features and data, our results on the training set go down and the test set values, in this case, look to converge.   

##Stepwise Regression

While stepwise regression has its many [critics](http://andrewgelman.com/2014/06/02/hate-stepwise-regression/), it is a useful exercise to introduce the concept of feature selection in the context of linear regression. This extra credit exercise has two components of different difficulties. First, use the `scikit-learn` reverse feature elimation (a greedy feature elimination algorithm) to implement something similar to sequential backward selection. The second, more difficult part is implementing sequential forward selection.


We generate a random dataset that has 100 features, but only 5 of them influence the response variable.  We can use sklearn's RFE to fit and attempt to rank the features.  


    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=5000, n_features=100, random_state=0)
    from sklearn.feature_selection import RFE
    linear = LinearRegression()
    rfe = RFE(linear,1)
    rfe.fit(X,y)
    rfe.ranking_




    array([  3,   2,  71,   1,   4,  47,  22,  36,  94,  61,  14,  39,  24,
            58,  20,  75,  98,  79,  41,  32,  78,  54,  76,  86,  55,  43,
            29,  66,  35,  68,  37,   8,  92,  51,  13,  89,  46,  82,  83,
            30,  96,  77,  31,  72,  90,  62,  67,  59,  73,  15,  11,  85,
            91,  17,  88,  26,  56,  28,  33,  53,  18,  10,   7,   6,  38,
            48,  50,  19,  84,  34,  93,  23,  27,  63,  70,  16,  12, 100,
            44,  80,  45,  97,  87,   9,  21,   5,  52,  60,  57,  49,  25,
            40,  95,  81,  42,  99,  74,  65,  69,  64])



We are going to loop through these variables and use them as a fit.   We want to see if a there is a point where adding more features will not improve the model.  


    linear = LinearRegression()
    rsq = []
    for i in range(2,50):
        linear.fit(X[:,rfe.ranking_ < i],y)
        rsq.append(linear.score(X[:,rfe.ranking_ < i],y))
    plt.plot(range(2,50),rsq,lw=3,color='seagreen',alpha=0.5,label='R-Square')
    plt.xlabel("Number of Variables")
    plt.ylabel('R-Squared')
    plt.ylim([0,1])
    plt.axvline(x=5,linestyle='--',color='gray')
    plt.show()


![png](http://www.bryantravissmith.com/img/output_21_0.png)


Instead of using RFE to do backward selection, we can create a class that implements sequential forward selection, which involves starting with no variables in the model, testing the addition of each variable using a chosen model comparison criterion, adding the variable (if any) that improves the model the most, and repeating this process until none improves the model.

#### Reference

* [Stepwise Regression Procedure](https://onlinecourses.science.psu.edu/stat501/node/88)


    class ForwardRegression:
        
        def __init__(self,X,y):
            self.nFeatures = X.shape[1]
            self.nRow = X.shape[0]
            self.X = X
            self.y = y
            self.linear = LinearRegression()
            self.scores = []
            self.best_features = []
            self.best_model = False
        
        def next_best(self):
            if not self.best_model:
                max_rsq = 0
                best_index = -1
                for i in range(self.nFeatures):
                    if i not in self.best_features:
                        self.linear.fit(self.X[:,(self.best_features+[i])],self.y)
                        score = self.linear.score(self.X[:,(self.best_features+[i])],self.y)
                        if score > max_rsq:
                            max_rsq = score
                            best_index = i
                if best_index != -1:
                    if len(self.scores) > 1:
                        if (max_rsq-self.scores[-1])/self.scores[-1] > 0.01:
                            self.best_features.append(best_index)
                            self.scores.append(max_rsq)
                        else:
                            self.best_model = True
            
                    else:
                        self.best_features.append(best_index)
                        self.scores.append(max_rsq)
            
    
    fr = ForwardRegression(X,y)
    for i in range(5):
        fr.next_best()
    print fr.scores
    print fr.best_features
    print fr.best_model

    [0.32517954427506357, 0.49706338477081047, 0.64962046580083821, 0.74271747755945094]
    [3, 1, 0, 4]
    True



    rfe.ranking_




    array([  3,   2,  71,   1,   4,  47,  22,  36,  94,  61,  14,  39,  24,
            58,  20,  75,  98,  79,  41,  32,  78,  54,  76,  86,  55,  43,
            29,  66,  35,  68,  37,   8,  92,  51,  13,  89,  46,  82,  83,
            30,  96,  77,  31,  72,  90,  62,  67,  59,  73,  15,  11,  85,
            91,  17,  88,  26,  56,  28,  33,  53,  18,  10,   7,   6,  38,
            48,  50,  19,  84,  34,  93,  23,  27,  63,  70,  16,  12, 100,
            44,  80,  45,  97,  87,   9,  21,   5,  52,  60,  57,  49,  25,
            40,  95,  81,  42,  99,  74,  65,  69,  64])



This matches with the previous result for this example.

##Afternoon - Lasso and Ridge Regresson

The goal fo this exercise is to get a feel for the shrinkage that these models do for coefficients.   We will start with the Ridge Regression in sklearn on the diabetes dataset.  We did a basic fit to start. 


    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    
    diabetes = load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    XT = diabetes.data[150:]
    yT = diabetes.target[150:]
    fit = Ridge(alpha=5, normalize=True).fit(X, y)
    print "MSE: ", mean_squared_error(fit.predict(X),y)

    MSE:  4251.12234379


Now we will look at how the parameters change as we increasee alpha.


    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    k = X.shape[1]
    alphas = np.logspace(-2, 2)
    params = np.zeros((len(alphas), k))
    for i,a in enumerate(alphas):
        X_data = scaler.fit_transform(X)
        fit = Ridge(alpha=a, normalize=True).fit(X_data, y)
        params[i] = fit.coef_
    
    plt.figure(figsize=(14,6))
    for param in params.T:
        plt.plot(alphas, param)
    plt.show()


![png](http://www.bryantravissmith.com/img/output_28_0.png)


Given a large enough alpha/lambda, the coefficients go to zero   This is a property of the model, and we want to try to use this paramter to find the best model.  


    k = X.shape[1]
    alphas = np.logspace(-3,0,1000)
    mse_train = []
    mse_test = []
    for i,a in enumerate(alphas):
        X_data = scaler.fit_transform(X)
        fit = Ridge(alpha=a, normalize=True).fit(X_data, y)
        mse_train.append(mean_squared_error(y,fit.predict(X_data)))
        mse_test.append(mean_squared_error(yT,fit.predict(scaler.transform(XT))))
    plt.figure(figsize=(14,6))
    plt.plot(alphas, mse_train)
    plt.plot(alphas, mse_test,color='green')
    plt.show()
    
    
    print "Best Alpha: ", alphas[mse_test.index(min(mse_test))]
    alpha_ridge = alphas[mse_test.index(min(mse_test))]


![png](http://www.bryantravissmith.com/img/output_30_0.png)


    Best Alpha:  0.252582002696


Now we are going to do the same for Lasso Regresion


    k = X.shape[1]
    alphas = np.logspace(-2, 2)
    params = np.zeros((len(alphas), k))
    for i,a in enumerate(alphas):
        X_data = scaler.fit_transform(X)
        fit = Lasso(alpha=a, normalize=True).fit(X_data, y)
        params[i] = fit.coef_
    
    plt.figure(figsize=(14,6))
    for param in params.T:
        plt.plot(alphas, param)
    plt.show()


![png](http://www.bryantravissmith.com/img/output_32_0.png)


The Lasso drops the paramters to zero much more quickly than the Ridge Method.  This is because of the absolut value in the prior.  


    k = X.shape[1]
    alphas = np.logspace(-3,0,1000)
    mse_train = []
    mse_test = []
    for i,a in enumerate(alphas):
        X_data = scaler.fit_transform(X)
        fit = Lasso(alpha=a, normalize=True).fit(X_data, y)
        mse_train.append(mean_squared_error(y,fit.predict(X_data)))
        mse_test.append(mean_squared_error(yT,fit.predict(scaler.transform(XT))))
    plt.figure(figsize=(14,6))
    plt.plot(alphas, mse_train)
    plt.plot(alphas, mse_test,color='green')
    plt.show()
    
    print "Best Alpha: ", alphas[mse_test.index(min(mse_test))]
    alpha_lasso = alphas[mse_test.index(min(mse_test))]


![png](http://www.bryantravissmith.com/img/output_34_0.png)


    Best Alpha:  0.286059553518


We are going to compare the 3 models.


    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X_data,y)
    
    lasso = Lasso(alpha=alpha_lasso)
    lasso.fit(X_data,y)
    
    ols = LinearRegression()
    ols.fit(X_data,y)
    
    print "Ridge MSE:",mean_squared_error(yT,ridge.predict(scaler.transform(XT)))
    print "Lasso MSE:",mean_squared_error(yT,lasso.predict(scaler.transform(XT)))
    print "OLS MSE:",mean_squared_error(yT,ols.predict(scaler.transform(XT)))
    print " "
    print "Ridge,Lasso,OLS" 
    for i in range(len(ridge.coef_)):
        print round(ridge.coef_[i],2),round(lasso.coef_[i],2),round(ols.coef_[i],2)

    Ridge MSE: 3184.60043415
    Lasso MSE: 3160.04592755
    OLS MSE: 3190.98282981
     
    Ridge,Lasso,OLS
    -3.28 -2.95 -3.29
    -17.46 -17.04 -17.51
    20.52 20.39 20.55
    14.68 14.28 14.7
    2.45 -0.0 3.85
    -15.54 -12.27 -16.78
    -12.38 -11.89 -13.07
    5.94 4.43 5.84
    26.36 27.43 25.93
    4.47 4.34 4.44


The Lasso model on the lower MSE on the test set, and also dropped one of the variables in the model.   Over all its coefficients are lower, and this is consistent with what we saw in the previous plots.


    from sklearn.metrics import r2_score
    print "Ridge MSE:",r2_score(yT,ridge.predict(scaler.transform(XT)))
    print "Lasso MSE:",r2_score(yT,lasso.predict(scaler.transform(XT)))
    print "OLS MSE:",r2_score(yT,ols.predict(scaler.transform(XT)))

    Ridge MSE: 0.473709418009
    Lasso MSE: 0.477767322867
    OLS MSE: 0.472654656259


It is clear that this is far from the final story...


    
