Title: Galvanize - Week 04 - Day 2
Date: 2015-06-23 10:20
Modified: 2015-06-23 10:30
Category: Galvanize
Tags: data-science, galvanize, Random Forests, sklearn
Slug: galvanize-data-science-04-02
Authors: Bryan Smith
Summary: Today we covered random forests.

#Galvanize Immersive Data Science

##Week 4 - Day 2

Today's quiz a an online interview questions that involves picking a random value from a stream in order 1 memory use. 
After that was a lecture on Bootstrap Aggregate (Bagging) ML methods, focused on decisions trees, then the random forest algorithm.   We implemented a random forest using our decision trees we made from yesterday.

##Random Forest Class

    from DecisionTree import DecisionTree
    import numpy as np 

    class RandomForest(object):
        '''A Random Forest class'''

        def __init__(self, num_trees, num_features,):
            '''
               num_trees:  number of trees to create in the forest:
            num_features:  the number of features to consider when choosing the
                               best split for each node of the decision trees
            '''
            self.num_trees = num_trees
            self.num_features = num_features
            self.forest = None

        def fit(self, X, y):
            '''
            X:  two dimensional numpy array representing feature matrix
                    for test data
            y:  numpy array representing labels for test data
            '''
            self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], self.num_features)

        def build_forest(self, X, y, num_trees, num_samples, num_features):
            '''
            Return a list of num_trees DecisionTrees.
            '''
            forest = []
            size_feature = np.floor(np.sqrt(num_features))
            if num_features < len(X[0]):
                size_feature = num_features

            for i in range(num_trees):

                #features = np.random.choice(range(len(X[0,:])),size=size_feature,replace=False)
                dt = DecisionTree(num_features = num_features)
                indexes = np.random.choice(range(num_samples),size=num_samples)
                dt.fit(X[indexes,:],y[indexes])
                forest.append(dt)

            return forest

        def predict(self, X):
            '''
            Return a numpy array of the labels predicted for the given test data.
            '''
            predictions = self.forest[0].predict(X).reshape(X.shape[0],1)
            for i in range(1,len(self.forest)):
                predictions = np.hstack((predictions,self.forest[i].predict(X).reshape(X.shape[0],1)))

            final_predictions = []
            for x in predictions:
                labels, counts = np.unique(x,return_counts=True)
                final_predictions.append(labels[np.argmax(counts)])

            return final_predictions

        def score(self, X, y):
            '''
            Return the accuracy of the Random Forest for the given test data and
            labels.

            '''
            prediction = self.predict(X)
            return np.sum(prediction==y).astype(float)/len(y)


## Checking Our Random Forest


    %matplotlib inline
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from DecisionTree import DecisionTree
    
    df = pd.read_csv('../data/playgolf.csv')
    y = df.pop('Result').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    dt = DecisionTree()
    dt.fit(X_train, y_train)
    predicted_y = dt.predict(X_test)
    
    print dt

    0
      |-> overcast:
      |     Play
      |-> no overcast:
      |     1
      |     |-> < 71:
      |     |     Play
      |     |-> >= 71:
      |     |     2
      |     |     |-> < 80:
      |     |     |     Play
      |     |     |-> >= 80:
      |     |     |     Don't Play


Our Decision tree is still functioning.  Since the random forest depends on that I wanted to just show that it functions.  The idea is that if we have a weak classifier we can use it a large number of them to vote on a decisions and find, on average, a better answer.  


    from RandomForest import RandomForest
    
    rf = RandomForest(num_trees=10, num_features=2)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    print "score:", rf.score(X_test, y_test),y_predict,y_test


    score: 0.5 ['Play', 'Play', 'Play', 'Play'] ['Play' 'Play' "Don't Play" "Don't Play"]


The random forest, unsurprisingly, does not do very well on the golf data set.  It is very small, and the rules are complicated.   Lets try it on some congress data.

##Comparision with Sklearn


    congress = pd.read_csv("../data/congressional_voting.csv",header=None)
    congress = congress.replace('republican',1.0).replace('democrat',0.0).replace('y',1.0).replace('n',0.0).replace('?',-1.0)
    y = congress[0].values
    x = congress.loc[:,1:].values
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.5)
    
    cv_score = 0.
    skcv_score = 0.
    from sklearn.cross_validation import KFold
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    
    kf = KFold(len(y_train), n_folds=10)
    for train_index, test_index in kf:
        rf = RandomForest(num_trees=10, num_features=5)
        skrf = RandomForestClassifier(n_estimators=10,max_features=5)
        rf.fit(X_train[train_index], y_train[train_index])
        skrf.fit(X_train[train_index], y_train[train_index])
        cv_score += rf.score(X_train[test_index], y_train[test_index])/10.
        skcv_score += skrf.score(X_train[test_index], y_train[test_index])/10.
        
    print "                       CV Accuracy    Test Accuracy"
    print "My RF Scores:         ", cv_score, accuracy_score(y_test,rf.predict(X_test))
    print "My Sklearn RF Scores: ", skcv_score, accuracy_score(y_test,skrf.predict(X_test))


                           CV Accuracy    Test Accuracy
    My RF Scores:          0.958441558442 0.94495412844
    My Sklearn RF Scores:  0.958658008658 0.95871559633


Sklearn's random forest classifier and my random forest implementation give similar results in cross validation and on the test set.  On small datasets, it works in similar time.   

##Afternoon

The afternoon paired sprint involved using and exploring sklearn's random forest classifier on a cell phone plan dataset attempting to predict churn.


    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_curve
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
    df = pd.read_csv('../data/churn.csv')
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account Length</th>
      <th>Area Code</th>
      <th>Phone</th>
      <th>Int'l Plan</th>
      <th>VMail Plan</th>
      <th>VMail Message</th>
      <th>Day Mins</th>
      <th>Day Calls</th>
      <th>Day Charge</th>
      <th>...</th>
      <th>Eve Calls</th>
      <th>Eve Charge</th>
      <th>Night Mins</th>
      <th>Night Calls</th>
      <th>Night Charge</th>
      <th>Intl Mins</th>
      <th>Intl Calls</th>
      <th>Intl Charge</th>
      <th>CustServ Calls</th>
      <th>Churn?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>...</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>...</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>...</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>...</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>...</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False.</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




    for i in df.columns:
        print i, df[i].nunique()

    State 51
    Account Length 212
    Area Code 3
    Phone 3333
    Int'l Plan 2
    VMail Plan 2
    VMail Message 46
    Day Mins 1667
    Day Calls 119
    Day Charge 1667
    Eve Mins 1611
    Eve Calls 123
    Eve Charge 1440
    Night Mins 1591
    Night Calls 120
    Night Charge 933
    Intl Mins 162
    Intl Calls 21
    Intl Charge 162
    CustServ Calls 10
    Churn? 2


We are going to clean up the data a little.   Replace boolean type values with 1 or 0, and drop some information that will not work with the classifier. 


    df['Int\'l Plan']=np.where(df['Int\'l Plan']=='yes',1,0)
    df['VMail Plan']=np.where(df['VMail Plan']=='yes',1,0)
    df['Churn?']=np.where(df['Churn?']=='True.',1,0)
    df = df[[u'Account Length', u'Int\'l Plan', u'VMail Plan', u'VMail Message', u'Day Mins', u'Day Calls', u'Day Charge', u'Eve Mins', u'Eve Calls', u'Eve Charge', u'Night Mins', u'Night Calls', u'Night Charge', u'Intl Mins', u'Intl Calls', u'Intl Charge', u'CustServ Calls', u'Churn?']]
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account Length</th>
      <th>Int'l Plan</th>
      <th>VMail Plan</th>
      <th>VMail Message</th>
      <th>Day Mins</th>
      <th>Day Calls</th>
      <th>Day Charge</th>
      <th>Eve Mins</th>
      <th>Eve Calls</th>
      <th>Eve Charge</th>
      <th>Night Mins</th>
      <th>Night Calls</th>
      <th>Night Charge</th>
      <th>Intl Mins</th>
      <th>Intl Calls</th>
      <th>Intl Charge</th>
      <th>CustServ Calls</th>
      <th>Churn?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>107</td>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>137</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    df.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3333 entries, 0 to 3332
    Data columns (total 18 columns):
    Account Length    3333 non-null int64
    Int'l Plan        3333 non-null int64
    VMail Plan        3333 non-null int64
    VMail Message     3333 non-null int64
    Day Mins          3333 non-null float64
    Day Calls         3333 non-null int64
    Day Charge        3333 non-null float64
    Eve Mins          3333 non-null float64
    Eve Calls         3333 non-null int64
    Eve Charge        3333 non-null float64
    Night Mins        3333 non-null float64
    Night Calls       3333 non-null int64
    Night Charge      3333 non-null float64
    Intl Mins         3333 non-null float64
    Intl Calls        3333 non-null int64
    Intl Charge       3333 non-null float64
    CustServ Calls    3333 non-null int64
    Churn?            3333 non-null int64
    dtypes: float64(8), int64(10)
    memory usage: 494.7 KB


So now we have a clean dataset with only ints and floats. Lets prepare our test and training


    y = df['Churn?'].values
    x = df.drop('Churn?',axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print "Base Accuracy: ", model.score(x_test,y_test)
    print "Guessing No Churn:", np.sum(y_test==0).astype(float)/len(y_test)

    Base Accuracy:  0.939
    Guessing No Churn: 0.862


The random forest is giving a better than guessing result!


    print confusion_matrix(y_test, model.predict(x_test))

    [[853   9]
     [ 52  86]]



    print "Recall: ", recall_score(y_test,y_predict)
    print "Precision: ", precision_score(y_test,y_predict)

    Recall:  0.623188405797
    Precision:  0.905263157895


The model is having the worst problem predicting that people who do churn will churn.   Only ~60% of those that churn were predicted to do so.  There is room for imporovement in the model here.

Sklearn's random forest classifiery has an out of bag error estimate as well as a estimator of feature importance.  We are going to use these next.


    model_oob = RandomForestClassifier(oob_score=True)
    model_oob.fit(x_train, y_train)
    oob_score = model_oob.score(x_test, y_test)
    print 'Accuracy (old/new): ', model.score(x_test,y_test), oob_score
    
    print "OBB Estimate: ", model_oob.oob_score_
    
    y_predict_oob = model_oob.predict(x_test)
    
    print 'Precision (old, new)', precision_score(y_test, y_predict), precision_score(y_test, y_predict_oob)
    print 'Recall (old, new)', recall_score(y_test, y_predict), recall_score(y_test, y_predict_oob)

    Accuracy (old/new):  0.939 0.943
    OBB Estimate:  0.921560222889
    Precision (old, new) 0.905263157895 0.917525773196
    Recall (old, new) 0.623188405797 0.644927536232


The difference between the old and new model is not statistically significant.  They are just the natural variation in this model.   The OBB estimate is very close to the accuracy on the test set.  That is promissing.  Lets see if we can find the most important features:


    features = model.feature_importances_
    print df.columns[model.feature_importances_ >= sorted(features)[-5]]
    features = model_oob.feature_importances_
    print df.columns[model_oob.feature_importances_ >= sorted(features)[-5]]

    Index([u'Int'l Plan', u'Day Mins', u'Day Charge', u'Eve Charge', u'CustServ Calls'], dtype='object')
    Index([u'Int'l Plan', u'Day Mins', u'Day Charge', u'Eve Charge', u'CustServ Calls'], dtype='object')


We can see that the two models share 4 out of 5 of the same most important features.   They differ on Eve. Charge and Int'l Plan.  

Before we start selecting featuers, we want to make sure that we are seeing the best model. 


    def tree_accuracy(num_trees):    
        model = RandomForestClassifier(oob_score=True, n_estimators=num_trees)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        return accuracy
        
    treevalues1 = range(1, 1000, 100)
    values1 = []
    for v in treevalues1:
        values1.append(tree_accuracy(v))
        
    treevalues2 = range(1, 200, 10)
    values2 = []
    for v in treevalues2:
        values2.append(tree_accuracy(v))
    plt.figure(figsize=(14,8))
    plt.subplot(1,2,1)
    plt.plot(treevalues1, values1,lw=2,color='seagreen')
    plt.subplot(1,2,2)
    plt.plot(treevalues2, values2,lw=2,color='steelblue')
    plt.xlabel('Number of Trees')
    plt.ylabel('accuracy')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D2/output_23_0.png)


We can see after about 50 estimators the accuracy does not significantly improve.   We also see if there is a limit on the number of features to consider at each node.


    def tree_accuracy(num_features, num_trees=50):    
        model = RandomForestClassifier(oob_score=True, n_estimators=num_trees, max_features=num_features)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        return accuracy
    
    treevalues1 = range(1, 17, 1)
    values1 = []
    for v in treevalues1:
        values1.append(tree_accuracy(v))
        
    plt.figure(figsize=(14,8))
    plt.plot(treevalues1, values1,lw=2,color='seagreen')
    
    plt.xlabel('Number of Features')
    plt.ylabel('accuracy')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D2/output_25_0.png)


We can consider 4 features for each node and about 5 estimators while improving the results.

I am wondering how this compares to other models we have covered.


    plt.figure(figsize=(14,8))
    for model in [LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(),RandomForestClassifier(n_estimators=50,max_features=5)]:
        model.fit(x_train,y_train)
        print "MODEL: ", model.__class__.__name__
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:,1]
        print "Accuracy", accuracy_score(y_pred,y_test)
        print "Recall", recall_score(y_pred,y_test)
        print "Precision", precision_score(y_pred,y_test)
        print ""
        fpr,trp,thres = roc_curve(y_test,y_prob)
        plt.plot(fpr,trp,label=model.__class__.__name__)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=4)
    plt.show()

    MODEL:  LogisticRegression
    Accuracy 0.872
    Recall 0.647058823529
    Precision 0.159420289855
    
    MODEL:  DecisionTreeClassifier
    Accuracy 0.907
    Recall 0.659574468085
    Precision 0.673913043478
    
    MODEL:  KNeighborsClassifier
    Accuracy 0.886
    Recall 0.714285714286
    Precision 0.289855072464
    
    MODEL:  RandomForestClassifier
    Accuracy 0.96
    Recall 0.929824561404
    Precision 0.768115942029
    



![png](http://www.bryantravissmith.com/img/GW04D2/output_27_1.png)


The random forest out performs the other models.   We have a relativley high true positve for a small false positive rate.   If we wanted to avoid false positive predictions of churn, the random forest can be turned to have a tre positive rate between 70 and 80%.   

Now lets try to find the most important features in the data set.  We are going through each tree in our model and getting the feature importance of that model.  We then will average over all the importance estimates.


    from collections import defaultdict
    from itertools import izip
    
    
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(x,y)
    
    d = defaultdict(list)
    
    for tree in model.estimators_:
        findex = np.argsort(tree.feature_importances_)
        ordered_features = df.columns[findex]
        values = tree.feature_importances_[findex]
        for i,name in izip(values,ordered_features):
            d[name].append(i)
            
    features = []
    feature_means = []
    feature_stds = []
    for k,v in d.iteritems():
        vals = np.array(v)
        mean = vals.mean()
        std = vals.std()
        features.append(k)
        feature_means.append(mean)
        feature_stds.append(std)
        
    feature_means = np.array(feature_means)
    indexes = np.argsort(feature_means)
    feature_means = feature_means[indexes]
    feature_stds = np.array(feature_stds)[indexes]/np.sqrt(999)
    features = np.array(features)[indexes]
    
    ind = np.arange(len(features))  # the x locations for the groups
    width = 0.55       # the width of the bars
    
    fig, ax = plt.subplots(figsize=(14,8))
    plt.bar(np.arange(len(features)), feature_means, width, color='r', yerr=feature_stds)
    ax.set_ylabel('Mean Importance')
    ax.set_title('Feature Importance')
    ax.set_xticks(ind)
    ax.set_xticklabels( features,rotation=45 )
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D2/output_29_0.png)


So we see that the most important features, on average, are day charge, day mins, custserv calls, int'l plan, eve charge, and int'l calls. 

Lets retrain the dataset on this subseted data.


    df2 = df[[u'Int\'l Plan', u'Day Mins', u'Day Charge', u'Eve Mins', u'Eve Charge', u'CustServ Calls']]
    x = df2.values




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Int'l Plan</th>
      <th>Day Mins</th>
      <th>Day Charge</th>
      <th>Eve Mins</th>
      <th>Eve Charge</th>
      <th>CustServ Calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>265.1</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>16.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>161.6</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>16.62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>243.4</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>10.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>299.4</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>5.26</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>166.7</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>12.61</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)
    model = RandomForestClassifier(n_estimators=100,max_features=5)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print "Base Accuracy: ", model.score(x_test,y_test)
    print "Guessing No Churn:", np.sum(y_test==0).astype(float)/len(y_test)
    print "Recall: ", recall_score(y_test,y_predict)
    print "Precision: ", precision_score(y_test,y_predict)
    print confusion_matrix(y_test, model.predict(x_test))

    Base Accuracy:  0.944
    Guessing No Churn: 0.857
    Recall:  0.671328671329
    Precision:  0.914285714286
    [[848   9]
     [ 47  96]]


The similar model gives similar results as the more complicated model, lending credence to the idea these are the most influencial factors.   If a cell phone company wants to reduce churn then they need to deal with the number of mins and charges for customer plans.   This is where they will reduce churn.


    
