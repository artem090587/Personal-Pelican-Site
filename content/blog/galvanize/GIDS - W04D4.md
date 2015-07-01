Title: Galvanize - Week 04 - Day 4
Date: 2015-06-25 10:20
Modified: 2015-06-25 10:30
Category: Galvanize
Tags: data-science, galvanize, Boosting, AdaBoost, GradientBoosting, machines
Slug: galvanize-data-science-04-04
Authors: Bryan Smith
Summary: Today we covered Boosting.

#Galvanize Immersive Data Science

##Week 4 - Day 4

Our quiz today had to do with the birthday problem and another problem that involved two hunting hounds.  The question was if there are two hunting hounds that successfully track with a probability p, is the strategy of following both hounds if they go in the same direction on a fork in the round, otherwise randomly guessing, better then just following 1 hound?  

The probability of both hounds being successful and matching is $p^2$ and the probability of both hounds matching and being unsuccessful is $(1-p)^2$.   The probability of not matching is $2p(1-p)$, and half of each time the hunter will randomly pick correct.  The exepected odds of success is $p^2 + p(1-p) = p^2 + p - p^2 = p$, the same as following one hound.   When I first read the problem I was not expecting that solution.   I like when I see something I was not expected!

##Morning Boosting

This morning we discussed boosting, and our morning sprint was to predict the bosting house prices using boosting on regression classifieres in [sklearn](http://scikit-learn.org/stable/).   


    %matplotlib inline
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import train_test_split, cross_val_score
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble.partial_dependence import plot_partial_dependence
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    boston = load_boston()
    # House Prices
    y = boston.target
    # The other 13 features
    x = boston.data
    
    #train and test set
    x_trn,x_test,y_trn,y_test = train_test_split(x,y,test_size=.2)
    
    plt.figure(figsize=(14,8))
    for i in range(len(x[0])):
        plt.subplot(4,4,i+1)
        plt.plot(x[:,i],y,color='steelblue',marker='o',lw=0)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_1_0.png)


    


For this sprint we really are not concerned with the distributions, but I still like to plot.   We see the house prices targets vary from 5 to 50.


    pd.Series(y).describe()




    count    506.000000
    mean      22.532806
    std        9.197104
    min        5.000000
    25%       17.025000
    50%       21.200000
    75%       25.000000
    max       50.000000
    dtype: float64




    print "MSE From Average: ", np.sum(np.power(y-y.mean(),2))/len(y)

    MSE From Average:  84.4195561562


Now that we have some baselines, we can now start to train our regressors and compare thier performance.  For this first trail I am going to make a Random Forest, GradientBoostingRegressor, and a AdaBoostRegressor.   I will be comparing the cross validated MSE and $r^2$ on the training set.


    rf = RandomForestRegressor(n_estimators=100,
                               n_jobs=-1,
                               random_state=1)
    
    gdbr = GradientBoostingRegressor(learning_rate=.1,
                                     loss='ls',
                                     n_estimators=100,
                                     random_state=1)
    
    abr = AdaBoostRegressor(DecisionTreeRegressor(),
                            learning_rate=0.1,
                            loss='linear',
                            n_estimators=100,
                            random_state=1)
    
    def mse(m,x,y):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(m.predict(x),y)
    
    def r2(m,x,y):
        from sklearn.metrics import r2_score
        return 1-(np.sum(np.power(m.predict(x)-y,2))/np.sum(np.power(y-y.mean(),2)))
    
    print rf.__class__.__name__,cross_val_score(rf,x_trn,y_trn,scoring=mse,cv=10).mean(),cross_val_score(rf,x_trn,y_trn,scoring=r2,cv=10).mean()
    print gdbr.__class__.__name__,cross_val_score(gdbr,x_trn,y_trn,scoring=mse,cv=10).mean(),cross_val_score(gdbr,x_trn,y_trn,scoring=r2,cv=10).mean()
    print abr.__class__.__name__,cross_val_score(abr,x_trn,y_trn,scoring=mse,cv=10).mean(),cross_val_score(abr,x_trn,y_trn,scoring=r2,cv=10).mean()
    


    RandomForestRegressor 11.6404925079 0.848311635198
    GradientBoostingRegressor 9.78445216743 0.871926100237
    AdaBoostRegressor 11.757802439 0.846590661377


The 10 fold cross validation on the training set gives MSE of order 10, much smaller than the naive estimate of 80.   All three of these models are doing well on the training set.   I would not pick one model over until I have tested them on the hold out set.  We can plot the performance of these models as we trained them.


    def stage_score_plot(model, train_x, train_y, test_x, test_y):
        from sklearn.metrics import mean_squared_error
        model.fit(train_x,train_y)
        mse_train = [mean_squared_error(train_y,yy) for yy in model.staged_predict(train_x)]
        mse_test = [mean_squared_error(test_y,yy) for yy in model.staged_predict(test_x)]
        xx = range(1,len(mse_test)+1)
        label = model.__class__.__name__ + " {} - Learning Rate " + str(model.learning_rate)
        plt.plot(xx,mse_train,lw=2,alpha=0.7,label=label.format("Train"))
        plt.plot(xx,mse_test,lw=2,alpha=0.7,label=label.format("Test"))
        plt.xlabel("Number of Iterations")
        plt.ylabel("Mean Square Error")
        
    plt.figure(figsize=(14,10))
    stage_score_plot(gdbr, x_trn, y_trn, x_test, y_test)
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_8_0.png)


Looking at the Gradient Boosting Regressor we see that as we add more weak learners/iterations, the training and test error drop together.  The Training error is still dropping, but the test error has leveled off around 8.  This result is also affected by the learning rate.  If we change it we get different results.


    plt.figure(figsize=(14,10))
    stage_score_plot(GradientBoostingRegressor(learning_rate=.1,loss='ls', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    stage_score_plot(GradientBoostingRegressor(learning_rate=1,loss='ls', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_10_0.png)


The higher learning rate leads to over fitting.   The training error goes to zero almost immediately, but the error on the test set is very high.

We can also lower the learning rate.  


    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    stage_score_plot(GradientBoostingRegressor(learning_rate=.1,loss='ls', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    stage_score_plot(GradientBoostingRegressor(learning_rate=.01,loss='ls', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    plt.legend()
    plt.subplot(1,2,2)
    stage_score_plot(GradientBoostingRegressor(learning_rate=.1,loss='ls', n_estimators=1000, random_state=1), x_trn, y_trn, x_test, y_test)
    stage_score_plot(GradientBoostingRegressor(learning_rate=.01,loss='ls', n_estimators=1000, random_state=1), x_trn, y_trn, x_test, y_test)
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_12_0.png)


In this case we see that the test error levels off to the same place for these two rates, but the lower learning rate takes more iterations to get there.

We can also compare the results of the gradient boosting to the random forest algorithm.


    plt.figure(figsize=(14,10))
    stage_score_plot(GradientBoostingRegressor(learning_rate=.1,loss='ls', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    plt.axhline(y=mean_squared_error(rf.fit(x_trn,y_trn).predict(x_test),y_test),color='orange',lw=3,linestyle='--',label='Random Forest Test')
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_15_0.png)


The random forest does not have the stage predict function that allows you to retroactively calculate the predictions at each stage of the training.   The end result is that for the same number of estimators/iterations, the Gradient Boosting Regressor does better on the Boston dataset.  

We can also look at the AdaBoostingRegessor because it does have stage predict.


    plt.figure(figsize=(14,10))
    plt.subplot(1,2,1)
    stage_score_plot(AdaBoostRegressor(learning_rate=1,loss='linear', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    stage_score_plot(AdaBoostRegressor(learning_rate=.1,loss='linear', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    #stage_score_plot(AdaBoostRegressor(learning_rate=.01,loss='linear', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    plt.axhline(y=mean_squared_error(rf.fit(x_trn,y_trn).predict(x_test),y_test),color='orange',linestyle='--',label='Random Forest')
    plt.legend()
    plt.subplot(1,2,2)
    stage_score_plot(AdaBoostRegressor(learning_rate=1,loss='linear', n_estimators=1000, random_state=1), x_trn, y_trn, x_test, y_test)
    stage_score_plot(AdaBoostRegressor(learning_rate=.1,loss='linear', n_estimators=1000, random_state=1), x_trn, y_trn, x_test, y_test)
    #stage_score_plot(AdaBoostRegressor(learning_rate=.01,loss='linear', n_estimators=100, random_state=1), x_trn, y_trn, x_test, y_test)
    plt.axhline(y=mean_squared_error(rf.fit(x_trn,y_trn).predict(x_test),y_test),color='orange',linestyle='--',label='Random Forest')
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_17_0.png)


In this case the AdaBoostRegressor does not do better than RandomForest on the test set.  Even allowing for more iterations (which takes a fair amoutn of time to fit).   We are using the naive parameters to fit the model.  We should really search for the best parameters in each model.

##Grid Search

The goal of grid searching is to fit the model using different parameters, and choose the result that has the best cross validated score.   This is not guaranteed to give the best results, but currently I do not know a better way to tune a model.  


    random_forest_grid = {'max_depth': [3, None],
                          'max_features': ['sqrt', 'log2', None],
                          'min_samples_split': [1, 2, 4],
                          'min_samples_leaf': [1, 2, 4],
                          'bootstrap': [True, False],
                          'n_estimators': [40, 80, 160, 320],
                          'random_state': [1]}
    
    rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                                 random_forest_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='mean_squared_error')
    rf_gridsearch.fit(x_trn, y_trn)
    
    print "best parameters:", rf_gridsearch.best_params_
    print "best score:",rf_gridsearch.best_score_
    best_rf_model = rf_gridsearch.best_estimator_


    Fitting 3 folds for each of 432 candidates, totalling 1296 fits


    [Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    0.1s
    [Parallel(n_jobs=-1)]: Done  50 jobs       | elapsed:    1.9s
    [Parallel(n_jobs=-1)]: Done 200 jobs       | elapsed:    8.2s
    [Parallel(n_jobs=-1)]: Done 450 jobs       | elapsed:   19.9s
    [Parallel(n_jobs=-1)]: Done 800 jobs       | elapsed:   42.4s
    [Parallel(n_jobs=-1)]: Done 1250 jobs       | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 1296 out of 1296 | elapsed:  1.3min finished


    best parameters: {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 40, 'min_samples_split': 1, 'random_state': 1, 'max_features': 'sqrt', 'max_depth': None}
    best score: -14.0843113088


You can see the MSE is negative, but that is an artifact of the fit used by sklearn.  The MSE is just the absolute value of that parameter.   We can use the best model from the search and get a feel for the results on the test set.


    mean_squared_error(best_rf_model.predict(x_test),y_test),mean_squared_error(RandomForestRegressor().fit(x_trn,y_trn).predict(x_test),y_test)




    (12.737843504901965, 9.5049568627450984)



Our tuned random forest did worse on the test set than our untuned model.  We would need to estimate the uncertainty of the MSE of both classifiers to get a feel for if this is a statisically significant difference. 


    gb_grid = {'learning_rate': [1,0.1,0.01],
                          'max_depth': [2,4,6],
                          'min_samples_leaf': [1, 2, 4],
                          'n_estimators': [20, 40, 80, 160],
                          'max_features': ['sqrt','log2',None],
                          'random_state': [1]}
    
    gb_gridsearch = GridSearchCV(GradientBoostingRegressor(),
                                 gb_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='mean_squared_error')
    gb_gridsearch.fit(x_trn, y_trn)
    
    print "best parameters:", gb_gridsearch.best_params_
    print "best score:", gb_gridsearch.best_score_
    best_gb_model = gb_gridsearch.best_estimator_


    Fitting 3 folds for each of 324 candidates, totalling 972 fits


    [Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=-1)]: Done  50 jobs       | elapsed:    0.3s
    [Parallel(n_jobs=-1)]: Done 200 jobs       | elapsed:    1.8s
    [Parallel(n_jobs=-1)]: Done 450 jobs       | elapsed:    4.7s
    [Parallel(n_jobs=-1)]: Done 800 jobs       | elapsed:   10.7s
    [Parallel(n_jobs=-1)]: Done 972 out of 972 | elapsed:   13.9s finished


    best parameters: {'learning_rate': 0.1, 'min_samples_leaf': 4, 'n_estimators': 160, 'random_state': 1, 'max_features': 'sqrt', 'max_depth': 4}
    best score: -13.2445739497



    mean_squared_error(best_gb_model.predict(x_test),y_test),mean_squared_error(GradientBoostingRegressor().fit(x_trn,y_trn).predict(x_test),y_test)




    (10.363886409362564, 6.8593273937564954)



We see a similar result in the Gradient Boosting Regressor.


    ada_grid = {'base_estimator': [best_gb_model,best_rf_model],
                'learning_rate': [1,0.1,0.01],
                'n_estimators': [20, 40, 80, 160],
                'random_state': [1]}
    
    ada_gridsearch = GridSearchCV(AdaBoostRegressor(),
                                 ada_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='mean_squared_error')
    ada_gridsearch.fit(x_trn, y_trn)
    
    print "best parameters:", ada_gridsearch.best_params_
    print "best score:", ada_gridsearch.best_score_
    best_ada_model = ada_gridsearch.best_estimator_


    Fitting 3 folds for each of 24 candidates, totalling 72 fits


    [Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    1.6s
    [Parallel(n_jobs=-1)]: Done  50 jobs       | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done  72 out of  72 | elapsed:  1.6min finished


    best parameters: {'n_estimators': 20, 'base_estimator': GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='ls',
                 max_depth=4, max_features='sqrt', max_leaf_nodes=None,
                 min_samples_leaf=4, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=160,
                 random_state=1, subsample=1.0, verbose=0, warm_start=False), 'random_state': 1, 'learning_rate': 0.1}
    best score: -12.8839320317



    mean_squared_error(best_ada_model.predict(x_test),y_test),mean_squared_error(AdaBoostRegressor().fit(x_trn,y_trn).predict(x_test),y_test)




    (11.267204239234372, 12.438897032159721)



The AdaBoostRegressor did improve over the default values, but in the end they all gave around the same MSE on the test set using the Bosting Housing Data.  

#Afternoon - AdaBoost

We started out the afternoon buliding our own AdaBoost Classification Algorithm, then we explored using sklearn's implementation to explore partial dependency plots.  


    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.base import clone
    
    
    class AdaBoostBinaryClassifier(object):
        '''
        INPUT:
        - n_estimator (int)
          * The number of estimators to use in boosting
          * Default: 50
    
        - learning_rate (float)
          * Determines how fast the error would shrink
          * Lower learning rate means more accurate decision boundary,
            but slower to converge
          * Default: 1
        '''
    
        def __init__(self,
                     n_estimators=50,
                     learning_rate=1):
    
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
            self.n_estimator = n_estimators
            self.learning_rate = learning_rate
    
            # Will be filled-in in the fit() step
            self.estimators_ = []
            self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)
    
        def fit(self, x, y):
            '''
            INPUT:
            - x: 2d numpy array, feature matrix
            - y: numpy array, labels
    
            Build the estimators for the AdaBoost estimator.
            '''
            w = np.ones(len(y)).astype(float)/len(y)
            
            for i in range(self.n_estimator):
                estimator, w, alpha = self._boost(x,y,w)
                self.estimators_.append(estimator)
                self.estimator_weight_[i] = alpha
    
    
        def _I(self,y1,y2):
            temp = (y1!=y2).astype(int)*2-1
            #print temp
            #return temp
    
        def _boost(self, x, y, sample_weight):
            '''
            INPUT:
            - x: 2d numpy array, feature matrix
            - y: numpy array, labels
            - sample_weight: numpy array
    
            OUTPUT:
            - estimator: DecisionTreeClassifier
            - sample_weight: numpy array (updated weights)
            - estimator_weight: float (weight of estimator)
    
            Go through one iteration of the AdaBoost algorithm. Build one estimator.
            '''
    
            estimator = clone(self.base_estimator)
            estimator.fit(x,y,sample_weight=sample_weight)
            ypred = estimator.predict(x)
            err_m = np.sum(sample_weight*(ypred!=y))/np.sum(sample_weight)
            alpha = np.log((1-err_m)/err_m)
            
            
            yp = 2*ypred-1
            yy = 2*y-1
            weights = sample_weight*np.exp(alpha*(ypred!=y))
            return estimator, weights, alpha
    
    
        def predict(self, x):
            '''
            INPUT:
            - x: 2d numpy array, feature matrix
    
            OUTPUT:
            - labels: numpy array of predictions (0 or 1)
            '''
            pred = np.zeros((1,x.shape[0]))
    
            for i,estimator in enumerate(self.estimators_):
                pred += self.estimator_weight_[i]*(2*estimator.predict(x)-1)
                #pred += self.estimator_weight_[i]*estimator.predict(x)
    
            pred = pred/np.abs(pred)
            pred = (pred+1)/2
            print pred
            return pred.astype(int)
    
    
    
        def score(self, x, y):
            '''
            INPUT:
            - x: 2d numpy array, feature matrix
            - y: numpy array, labels
    
            OUTPUT:
            - score: float (accuracy score between 0 and 1)
            '''
            return np.sum(self.predict(x)==y).astype(float)/len(y)


The above is our AdaBoost Classifory class.  We will be using it on spam data.


    data = np.genfromtxt('boosting/data/spam.csv', delimiter=',')
    y = data[:, -1]
    x = data[:, 0:-1]
    train_x, test_x, train_y, test_y = train_test_split(x, y)
    
    my_ada = AdaBoostBinaryClassifier(n_estimators=50)
    my_ada.fit(train_x, train_y)
    print "Accuracy:", my_ada.score(test_x, test_y)

    Accuracy: [[ 0.  0.  1. ...,  1.  1.  1.]]
    0.917463075586


Our out of the box score is around 92% accuracy.   We will be exploring feature importance using sklearn's implementation, so I will read in the naems from the clipboard.


    df = pd.read_clipboard()
    names = df.values[:,0]
    names




    array(['word_freq_make:', 'word_freq_address:', 'word_freq_all:',
           'word_freq_3d:', 'word_freq_our:', 'word_freq_over:',
           'word_freq_remove:', 'word_freq_internet:', 'word_freq_order:',
           'word_freq_mail:', 'word_freq_receive:', 'word_freq_will:',
           'word_freq_people:', 'word_freq_report:', 'word_freq_addresses:',
           'word_freq_free:', 'word_freq_business:', 'word_freq_email:',
           'word_freq_you:', 'word_freq_credit:', 'word_freq_your:',
           'word_freq_font:', 'word_freq_000:', 'word_freq_money:',
           'word_freq_hp:', 'word_freq_hpl:', 'word_freq_george:',
           'word_freq_650:', 'word_freq_lab:', 'word_freq_labs:',
           'word_freq_telnet:', 'word_freq_857:', 'word_freq_data:',
           'word_freq_415:', 'word_freq_85:', 'word_freq_technology:',
           'word_freq_1999:', 'word_freq_parts:', 'word_freq_pm:',
           'word_freq_direct:', 'word_freq_cs:', 'word_freq_meeting:',
           'word_freq_original:', 'word_freq_project:', 'word_freq_re:',
           'word_freq_edu:', 'word_freq_table:', 'word_freq_conference:',
           'char_freq_;:', 'char_freq_(:', 'char_freq_[:', 'char_freq_!:',
           'char_freq_$:', 'char_freq_#:', 'capital_run_length_average:',
           'capital_run_length_longest:', 'capital_run_length_total:'], dtype=object)



To explore this we want to get a feel for the misclassification error of databoost.  We redefined our plot function to score misclassification instead of MSE.


    %matplotlib inline
    import matplotlib.pyplot as plt
    
    def stage_score_plot(model, train_x, train_y, test_x, test_y):
        from sklearn.metrics import accuracy_score
        model.fit(train_x,train_y)
        acc_train = [1-accuracy_score(train_y,yy) for yy in model.staged_predict(train_x)]
        acc_test = [1-accuracy_score(test_y,yy) for yy in model.staged_predict(test_x)]
        xx = range(1,len(acc_test)+1)
        label = model.__class__.__name__ + " {} - Learning Rate " + str(model.learning_rate)
        plt.plot(xx,acc_train,lw=2,alpha=0.7,label=label.format("Train"))
        plt.plot(xx,acc_test,lw=2,alpha=0.7,label=label.format("Test"))
        plt.xlabel("Number of Iterations")
        plt.ylabel("Misclassification")
        
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    plt.figure(figsize=(14,8))
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)
    stage_score_plot(clf,train_x, train_y, test_x, test_y)
    
    clf = GradientBoostingClassifier(n_estimators=100)
    stage_score_plot(clf,train_x, train_y, test_x, test_y)
    
    clf = GradientBoostingClassifier(n_estimators=100,max_depth=3)
    stage_score_plot(clf,train_x, train_y, test_x, test_y)
    
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(14,8))
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)
    stage_score_plot(clf,train_x, train_y, test_x, test_y)
    
    clf = GradientBoostingClassifier(n_estimators=100,max_depth=10)
    stage_score_plot(clf,train_x, train_y, test_x, test_y)
    
    clf = GradientBoostingClassifier(n_estimators=100,max_depth=100)
    stage_score_plot(clf,train_x, train_y, test_x, test_y)
    
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_36_0.png)



![png](http://www.bryantravissmith.com/img/GW04D4/output_36_1.png)


The top plot shows AdaBoost vs GradientBoosting with differnt max depths set.   The Ada performas better on the test set.  As we increase the depth of the trees used in the GradientBoosting, we see in the bottom plot the over fitting is abundant.   We need to have week learners to get optimal results with this method.  A strong learner will still overfit the data when boosted.


    from sklearn.grid_search import GridSearchCV
    
    gb_grid = {'learning_rate' : [0.01, 0.1, 1],
               'n_estimators' : [50,100,150,200],
               'max_depth' : [2,4,6],
               'max_features': ['sqrt', 'log2', None],
               'random_state': [1]}
    
    gb_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                 gb_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='accuracy')
    gb_gridsearch.fit(train_x, train_y)
    
    print "best parameters:", gb_gridsearch.best_params_
    print "best score:", gb_gridsearch.best_score_
    
    best_gb_model = gb_gridsearch.best_estimator_
    from sklearn.metrics import accuracy_score
    accuracy_score(best_gb_model.predict(test_x),test_y)

    Fitting 3 folds for each of 108 candidates, totalling 324 fits


    [Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    0.2s
    [Parallel(n_jobs=-1)]: Done  50 jobs       | elapsed:    8.1s
    [Parallel(n_jobs=-1)]: Done 200 jobs       | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done 318 out of 324 | elapsed:  1.5min remaining:    1.7s
    [Parallel(n_jobs=-1)]: Done 324 out of 324 | elapsed:  1.6min finished


    best parameters: {'max_features': 'log2', 'n_estimators': 200, 'learning_rate': 0.1, 'random_state': 1, 'max_depth': 6}
    best score: 0.952463768116





    0.95221546481320596



This is a good improvement over the previous model.   With GradientBoosting, we cal now explore which features are the most important features for identifying spam.

##Feature Importance


    import matplotlib.pyplot as plt
    import numpy as np
    
    indexes = np.argsort(best_gb_model.feature_importances_)
    
    fig = plt.figure(figsize=(14, 18))
    x_ind = np.arange(best_gb_model.feature_importances_.shape[0])
    plt.barh(x_ind, best_gb_model.feature_importances_[indexes], height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, names[indexes], fontsize=8)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_40_0.png)


We see that use of excessive capitalizaiton and expclaimations are important for predicting spam.   The use of 'cs' or 'telnet' are not.   The use of a partial dependency plot allows us to get a feel for how changing the values of these features affect the outcome of the predictions.


    from sklearn.ensemble.partial_dependence import plot_partial_dependence
    from sklearn.ensemble.partial_dependence import partial_dependence
    
    plt.figure(figsize=(14,14))
    ax = plt.gca()
    for i in indexes[:12]:
        result = partial_dependence(best_gb_model, i, X=train_x, grid_resolution=50)
    
        plt.plot((result[1][0])/(result[1][0].max()-result[1][0].min()),result[0][0],label=names[i])
        #plot_partial_dependence(best_gb_model, train_x,indexes[:12],feature_names=names,n_jobs=3, grid_resolution=50,ax=ax)
    plt.legend(bbox_to_anchor=[1.25,1.005])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_42_0.png)


As we increase the freq_lab or freq_address, we have a incrasing and decreasing accuracy of spam classification.  The parital dependency on these features is high.  Lab and telenet are also more than most, which is interesting because telent is low feature importance. 

It is possible to make 2D plots and 3D plots of partial dependancies.


    from sklearn.ensemble.partial_dependence import partial_dependence
    
    couple_of_tuples = [(x,y) for x in indexes[:4] for y in indexes[4:6]]
    plot_partial_dependence(best_gb_model, train_x,couple_of_tuples,feature_names=names,n_jobs=3, grid_resolution=50,figsize=(14,14))
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D4/output_44_0.png)


From these plots we can see that word-frequency has co-dependency with freq data and freq_parts 


    
