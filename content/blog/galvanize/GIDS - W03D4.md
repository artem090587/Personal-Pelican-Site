Title: Galvanize - Week 03 - Day 4
Date: 2015-06-18 10:20
Modified: 2015-06-18 10:30
Category: Galvanize
Tags: data-science, galvanize, Lasso Regresion, Ridge Regression, Regularization
Slug: galvanize-data-science-03-04
Authors: Bryan Smith
Summary: Today we covered logistic regression and ROC curves.

#Galvanize Immersive Data Science

##Week 3 - Day 4

Today we had a 2 hour assessment on everything we covered.  There were programming style problem and 1 math style problems.  The topics were on everything we have covered up to now.  

## Afternoon

The lectures today were on Logistic Regresion, Odds, and ROC Curves.   


##ROC Curve 

We were told that one of the best ways to evaluate how a classifier performs is an [ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic).   They display the change in the false and true positive rates as paramters in the model change.  In the case of logistic regression, its the threshold use to classify a data point based on the predicted probability. 

Recall that the *true positive rate* is

```
 number of true positives     number correctly predicted positive
-------------------------- = -------------------------------------
 number of positive cases           number of positive cases
```

and the *false positive rate* is

```
 number of false positives     number incorrectly predicted positive
--------------------------- = ---------------------------------------
  number of negative cases           number of negative cases
```


    %matplotlib inline
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from __future__ import division
    
    def roc_curve(p,y):
        thresholds = np.linspace(0,1,100)
        TPR = []
        FPR = []
        for thresh in thresholds:
            pred = (p>=thresh).astype(int)
            TPR.append(np.sum((pred==y)&(y==1))/np.sum(y))
            FPR.append(np.sum((pred!=y)&(y==0))/np.sum(y==0))
        TPR = np.array(TPR)
        FPR = np.array(FPR)
        return TPR,FPR,thresholds
        
    
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=2, n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    tpr, fpr, thresholds = roc_curve(probabilities, y_test)
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of fake data")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D4/output_1_0.png)



    from sklearn.metrics import roc_curve as roc
    fpr1,tpr1,threshs = roc(y_test,probabilities)
    plt.plot(fpr, tpr,color='seagreen',linestyle='--',marker='o',alpha=0.8,lw=3,label='Bryans ROC')
    plt.plot(fpr1, tpr1,color='indianred',alpha=0.8,lw=3,label='Sklearn ROC')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of fake data")
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D4/output_2_0.png)


The results between the two curves is negligable on a fake dataset.  We are now going to do this with FICO data.


    import pandas as pd
    df = pd.read_csv('data/loanf.csv')
    y = (df['Interest.Rate'] <= 12).values
    X = df[['FICO.Score', 'Loan.Length', 'Loan.Amount']].values


    from sklearn.cross_validation import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score
    
    a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model = LogisticRegression()
    cal = CalibratedClassifierCV(model, method='isotonic', cv=20)
    cal.fit(a_train, b_train)
    probs = cal.predict_proba(a_test)[:,1]
    tpr, fpr, thresholds = roc_curve(probs, b_test)
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of Loan Data")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    
    print "AUC: ", roc_auc_score(cal.predict(a_test),b_test)
    print "Accuracy: ", accuracy_score(cal.predict(a_test),b_test)
    print "Recall: ", recall_score(cal.predict(a_test),b_test)
    print "Precision: ", precision_score(cal.predict(a_test),b_test)


![png](http://www.bryantravissmith.com/img/GW03D4/output_5_0.png)


     AUC:  0.763176276353
    Accuracy:  0.769333333333
    Recall:  0.744094488189
    Precision:  0.636363636364


The the model is not a great model, but it does start off identifying 3 out of 4 true positives and has 6 our of 10 of those predicted to be positive actually being positive.  


##Graduate School Admissions

The data we will be using net is admission data on Grad school acceptances.

* `admit`: whether or not the applicant was admitted to grad. school
* `gpa`: undergraduate GPA
* `GRE`: score of GRE test
* `rank`: prestige of undergraduate school (1 is highest prestige, ala Harvard)

We will use the GPA, GRE, and rank of the applicants to try to predict whether or not they will be accepted into graduate school.

Before we get to predictions, we should do some data exploration.

1. Load in the dataset into pandas: `data/grad.csv`.


    grad = pd.read_csv('data/grad.csv')
    grad.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>380</td>
      <td>3.61</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>660</td>
      <td>3.67</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>800</td>
      <td>4.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>640</td>
      <td>3.19</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>520</td>
      <td>2.93</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




    grad.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 400 entries, 0 to 399
    Data columns (total 4 columns):
    admit    400 non-null int64
    gre      400 non-null int64
    gpa      400 non-null float64
    rank     400 non-null int64
    dtypes: float64(1), int64(3)
    memory usage: 15.6 KB



    grad.describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.317500</td>
      <td>587.700000</td>
      <td>3.389900</td>
      <td>2.48500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.466087</td>
      <td>115.516536</td>
      <td>0.380567</td>
      <td>0.94446</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>220.000000</td>
      <td>2.260000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>520.000000</td>
      <td>3.130000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>580.000000</td>
      <td>3.395000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>660.000000</td>
      <td>3.670000</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>800.000000</td>
      <td>4.000000</td>
      <td>4.00000</td>
    </tr>
  </tbody>
</table>
</div>




    temp = pd.crosstab(grad['admit'],grad['rank'])
    temp




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>rank</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>admit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28</td>
      <td>97</td>
      <td>93</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>54</td>
      <td>28</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




    temp.transpose().plot(kind='bar')




    <matplotlib.axes._subplots.AxesSubplot at 0x10a7a8a90>




![png](http://www.bryantravissmith.com/img/GW03D4/output_11_1.png)


We see that if a person is applying to grad school from a rank 1 school, they are more likely than not to be accepeted.   We also see that this ratio drops as the rank of the current school is lower.   We also see that most of the dat is from ranke 2 and rank 3 scores.


    (temp/temp.sum()).transpose().plot(kind='bar')




    <matplotlib.axes._subplots.AxesSubplot at 0x10f591a50>




![png](http://www.bryantravissmith.com/img/GW03D4/output_13_1.png)


Looking at the rations instead of hte counts highlight that change.   There is an increase in chances of getting accepted if the person is coming from a better school.   

Lets look at the GRE and GPA distributions.


    grad.gre.hist()




    <matplotlib.axes._subplots.AxesSubplot at 0x10f854990>




![png](http://www.bryantravissmith.com/img/GW03D4/output_15_1.png)



    grad.gpa.hist()




    <matplotlib.axes._subplots.AxesSubplot at 0x10f8f2a10>




![png](http://www.bryantravissmith.com/img/GW03D4/output_16_1.png)


Both of these are skewed left, but we do have a cut off on the GPA not being above 4.  We see a spike there, and this is commonly seen in any data with arbitary cutoffs.   If there was no max GPA, we would expenct that some people would have 5's,6's, and so on.  They might be rare, but they are there.  The cap compresses all these overacheivers to 4.0.  

##Fitting Grad School Admissions


    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
    y = grad[['admit']]
    X = sm.add_constant(grad[['gre','gpa','rank']])
    model = Logit(y,X).fit()
    model.summary()

    Optimization terminated successfully.
             Current function value: 0.574302
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>admit</td>      <th>  No. Observations:  </th>  <td>   400</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   396</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     3</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 19 Jun 2015</td> <th>  Pseudo R-squ.:     </th>  <td>0.08107</td> 
</tr>
<tr>
  <th>Time:</th>              <td>07:23:59</td>     <th>  Log-Likelihood:    </th> <td> -229.72</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -249.99</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>8.207e-09</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th> <td>   -3.4495</td> <td>    1.133</td> <td>   -3.045</td> <td> 0.002</td> <td>   -5.670    -1.229</td>
</tr>
<tr>
  <th>gre</th>   <td>    0.0023</td> <td>    0.001</td> <td>    2.101</td> <td> 0.036</td> <td>    0.000     0.004</td>
</tr>
<tr>
  <th>gpa</th>   <td>    0.7770</td> <td>    0.327</td> <td>    2.373</td> <td> 0.018</td> <td>    0.135     1.419</td>
</tr>
<tr>
  <th>rank</th>  <td>   -0.5600</td> <td>    0.127</td> <td>   -4.405</td> <td> 0.000</td> <td>   -0.809    -0.311</td>
</tr>
</table>



We see that the model is not very good.  The Pseudo R-square, which as to do with the deviance or negative log likelihood, is very low.  The coefficients are interested in that they do predict more likely admision for higher gre, higher gpa, and better schools.   

I think I want to plot the over of the predictions to show how poor the model is.


    xp = X.values[(y.values==1)[:,0],:]
    yp = model.fittedvalues.values[(y.values==1)[:,0]]
    xn = X.values[(y.values==0)[:,0],:]
    yn = model.fittedvalues.values[(y.values==0)[:,0]]
    zp = xp.dot(model.params.values)
    zn = xn.dot(model.params.values)
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(zp,np.exp(yp)/(1+np.exp(yp)),'go',alpha=0.5)
    plt.xlabel("Logodds for Datapoint")
    plt.ylabel("Prob for Student Admited")
    plt.subplot(1,2,2)
    plt.plot(zn,np.exp(yn)/(1+np.exp(yn)),'ro',alpha=0.5)
    plt.xlabel("Logodds for Datapoint")
    plt.ylabel("Prob for Student Not Admited")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D4/output_20_0.png)


This model does not do a great job of predicting admissions, but we can attemp to use Sklearn's machinery to get a better fit, and also measure the model metrics simply.




    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import accuracy_score,roc_auc_score
    
    
    a_train, a_test, b_train, b_test = train_test_split(grad[['gre','gpa','rank']].values, grad.admit.values, test_size=0.30, random_state=42)
    lin = LogisticRegression()
    cal1=CalibratedClassifierCV(lin, method='isotonic', cv=20)
    cal1.fit(a_train,b_train)
    lin = LogisticRegressionCV(cv=20,scoring='accuracy')
    lin.fit(a_train,b_train)
    probs = cal1.predict_proba(a_test)[:,1]
    tpr, fpr, thresholds = roc_curve(probs, b_test)
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of College Data")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    print "AUC: ", roc_auc_score(cal1.predict(a_test),b_test)
    print "Accuracy: ", accuracy_score(cal1.predict(a_test),b_test)
    print "Recall: ", recall_score(cal1.predict(a_test),b_test)
    print "Precision: ", precision_score(cal1.predict(a_test),b_test)


![png](http://www.bryantravissmith.com/img/GW03D4/output_22_0.png)


    AUC:  0.613445378151
    Accuracy:  0.666666666667
    Recall:  0.485714285714
    Precision:  0.435897435897


The area under the ROC curve is not far from 0.5 (random guessing).  The model only predicts 50% of the students admitted to college of being admitted, and only 43% of those predicted to be admitted were actually admitted.   I do not think the College Board will be breaking down our doors for this model.

In one way we treated the Rank as a continuous variable, and could try treating it like a categorical variable instead.


    grad1 = pd.get_dummies(grad, columns=['rank'])
    a_train, a_test, b_train, b_test = train_test_split(grad1.drop(['admit','rank_4'],axis=1).values, grad1.admit.values, test_size=0.30, random_state=42)
    
    lin = LogisticRegression()
    cal2=CalibratedClassifierCV(lin, method='isotonic', cv=20)
    cal2.fit(a_train,b_train)
    
    probs = cal2.predict_proba(a_test)[:,1]
    tpr1, fpr1, thresholds1 = roc_curve(probs, b_test)
    
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3,label="Original")
    plt.plot(fpr1, tpr1,color='indianred',alpha=0.8,lw=3,label="With Rank Categories")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of College Data")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc=4)
    plt.show()
    print "AUC: ", roc_auc_score(cal2.predict(a_test),b_test)
    print "Accuracy: ", accuracy_score(cal2.predict(a_test),b_test)
    print "Recall: ", recall_score(cal2.predict(a_test),b_test)
    print "Precision: ", precision_score(cal2.predict(a_test),b_test)


![png](http://www.bryantravissmith.com/img/GW03D4/output_24_0.png)


    AUC:  0.640625
    Accuracy:  0.691666666667
    Recall:  0.53125
    Precision:  0.435897435897


We could have an initial pass where we want a TPR > 60% and FPR < 40.  We can find the thresholds for these values.


    thresholds[(tpr > 0.6)&(fpr < 0.4)]




    array([ 0.24242424,  0.25252525,  0.26262626])



##Beta coefficients as Odds Ratio

One thing that is often lost when talking about logistic regression is the idea of the odds ratio, or rather the probabilistic interpretation of the model. For this next part we will get hands on with the odds ratio.

The ***odds ratio*** is defined as the product of the exponential of each coefficient.

![](images/odds_ratio.png)

This is the odds of being admitted over not being admitted.

It tells you how much a one unit increase of a feature corresponds to the odds of being admitted to grad school. And in doing so the coefficients of the logistic regression can be interpreted similarly to the coefficients of linear regression.

From our model we can look at the beta coefficients (Intercept,GRE, GPA, Rank)


    beta = cal1.calibrated_classifiers_[-1].__dict__['base_estimator'].coef_[0,:]
    print beta
    beta = np.hstack((cal1.calibrated_classifiers_[-1].__dict__['base_estimator'].intercept_[0],beta))
    beta

    [ 0.00084169  0.36208945 -0.54024057]





    array([ -1.19927209e+00,   8.41691796e-04,   3.62089453e-01,
            -5.40240569e-01])



The odd ratios for each beta is given by:

$$\mbox{odds ratio} = e^{\beta_i}$$


    np.exp(beta)




    array([ 0.30141353,  1.00084205,  1.43632742,  0.58260808])



This means that the base line odd ration of getting into grad school is 0.3.
    
**Increasing your gre by 1 point increases your odds by 1.0008.**

**Increasing your gpa by 1 point increases your ods by 1.435.**

**Decreasing your rank of college 1 unit decreases your odds by 0.582.**

We can also ask how much change would result in doubling the odds ratio:

$$x_i = \frac{ln(2)}{\beta_i}$$


    np.log(2)/beta




    array([ -5.77973244e-01,   8.23516617e+02,   1.91429818e+00,
            -1.28303430e+00])



Increasing my GRE by 824 doubles my odds ration.  Increasing my GPA by 1.91 doubles the odds ratio, and increasing the ranking by 1.28 doubles the odds ratio.  

##Predicted Probabilities

Now let's actually play with our data to verify what we calculated above with the Odds Ratio.  We can look, on average, how the rank changes the odds.


    g = grad.groupby('rank').mean().reset_index().drop('admit',axis=1)
    g




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>gre</th>
      <th>gpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>611.803279</td>
      <td>3.453115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>596.026490</td>
      <td>3.361656</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>574.876033</td>
      <td>3.432893</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>570.149254</td>
      <td>3.318358</td>
    </tr>
  </tbody>
</table>
</div>




    p_g = cal1.predict_proba(g[['gre','gpa','rank']].values)[:,1]
    p_g




    array([ 0.68469485,  0.3519206 ,  0.17594864,  0.05175475])



We can use this to calculate the odds rations for each rank


    odds_g = p_g/(1-p_g)
    final = np.vstack((np.arange(1,5),np.vstack((p_g,odds_g))))
    final




    array([[ 1.        ,  2.        ,  3.        ,  4.        ],
           [ 0.68469485,  0.3519206 ,  0.17594864,  0.05175475],
           [ 2.1715308 ,  0.54302082,  0.21351659,  0.0545795 ]])




    predicted_odds = np.hstack((np.array(2.1715408),(odds_g*0.583)[:3]))
    predicted_odds




    array([ 2.1715408 ,  1.26600246,  0.31658114,  0.12448017])



The odds to drop, but they do not match the vlaues we have from the average predictions.  We can make a graph of the log odds and find the slope:


    pd.DataFrame({"logodds":np.log(odds_g),"rank":range(1,5)}).set_index('rank').plot(kind='bar')
    print -1.7/3,beta[3]

    -0.566666666667 -0.540240569018



![png](http://www.bryantravissmith.com/img/GW03D4/output_41_1.png)


In this case the slop of the logodds and the fitted coefficient match up very close.  Lets do this for the GRE and GPA


    
    from sklearn.linear_model import LinearRegression
    g = grad.groupby('gre').mean().reset_index().drop('admit',axis=1)
    p_g = cal1.predict_proba(g[['gre','gpa','rank']].values)[:,1]
    odds_g = p_g/(1-p_g)
    plt.plot(g.gpa.values,odds_g,'go')
    linear = LinearRegression()
    linear.fit(g.gre.values.reshape(26,1),odds_g.reshape(26,1))
    print linear.coef_[0][0],beta[1]

    0.00138822870627 0.000841691795892



![png](http://www.bryantravissmith.com/img/GW03D4/output_43_1.png)


They are the same scale, but differ by a factor 1.5.   This makes sense because we have a few clear outliers in the data.


    
    from sklearn.linear_model import LinearRegression
    g = grad.groupby('gpa').mean().reset_index().drop('admit',axis=1)
    p_g = cal1.predict_proba(g[['gre','gpa','rank']].values)[:,1]
    odds_g = p_g/(1-p_g)
    plt.plot(g.gpa.values,odds_g,'go')
    linear = LinearRegression()
    #linear.fit(g.gpa.values,odds_g)
    linear.fit(g.gpa.values.reshape(132,1),odds_g.reshape(132,1))
    print linear.coef_[0][0],beta[2]

    0.506303444619 0.362089452847



![png](http://www.bryantravissmith.com/img/GW03D4/output_45_1.png)


This is also close, and the same order of magnitude.  But we have some clear outliers that need to be address in with this model.   The fact tha the coefficients of the model have a interpretation in terms of odds is great.  It adds a hook to reality in this abstraction. 

## MOOCS

This is the future!  No one goes to physical schools any more and MOOCs rule the world.

Harvard and MIT have [released](http://newsoffice.mit.edu/2014/mit-and-harvard-release-de-identified-learning-data-open-online-courses) a great dataset around engagement statistics for their MOOC courses. One of the biggest issues with MOOCs is engagement. We will try to predict the probability of 'engagement' of a student given all the other columns.  We will define engagement here as either: `explored == 1 OR certified == 1`.


    mooc = pd.read_csv('data/mooc.csv')
    mooc['engagement'] = mooc['explored']+mooc['certified']
    mooc['engagement'] = np.where(mooc.engagement > 0,1,0)
    mooc.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>course_id</th>
      <th>userid_DI</th>
      <th>registered</th>
      <th>viewed</th>
      <th>explored</th>
      <th>certified</th>
      <th>final_cc_cname_DI</th>
      <th>LoE_DI</th>
      <th>YoB</th>
      <th>gender</th>
      <th>...</th>
      <th>start_time_DI</th>
      <th>last_event_DI</th>
      <th>nevents</th>
      <th>ndays_act</th>
      <th>nplay_video</th>
      <th>nchapters</th>
      <th>nforum_posts</th>
      <th>roles</th>
      <th>incomplete_flag</th>
      <th>engagement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HarvardX/CB22x/2013_Spring</td>
      <td>MHxPC130442623</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2012-12-19</td>
      <td>2013-11-17</td>
      <td>NaN</td>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HarvardX/CS50x/2012</td>
      <td>MHxPC130442623</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HarvardX/CB22x/2013_Spring</td>
      <td>MHxPC130275857</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2013-02-08</td>
      <td>2013-11-17</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HarvardX/CS50x/2012</td>
      <td>MHxPC130275857</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2012-09-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HarvardX/ER22x/2013_Spring</td>
      <td>MHxPC130275857</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2012-12-19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




    mg = mooc.groupby('course_id')['course_id','viewed','explored','certified','engagement'].mean()
    mg = mg.sort(['engagement'])
    mg




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>viewed</th>
      <th>explored</th>
      <th>certified</th>
      <th>engagement</th>
    </tr>
    <tr>
      <th>course_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HarvardX/CB22x/2013_Spring</th>
      <td>0.543764</td>
      <td>0.018232</td>
      <td>0.012799</td>
      <td>0.018299</td>
    </tr>
    <tr>
      <th>MITx/3.091x/2013_Spring</th>
      <td>0.960906</td>
      <td>0.023457</td>
      <td>0.022479</td>
      <td>0.028506</td>
    </tr>
    <tr>
      <th>HarvardX/PH278x/2013_Spring</th>
      <td>0.379173</td>
      <td>0.030049</td>
      <td>0.017954</td>
      <td>0.031867</td>
    </tr>
    <tr>
      <th>MITx/6.002x/2013_Spring</th>
      <td>0.480549</td>
      <td>0.040612</td>
      <td>0.026670</td>
      <td>0.041511</td>
    </tr>
    <tr>
      <th>MITx/8.MReV/2013_Summer</th>
      <td>0.708663</td>
      <td>0.039675</td>
      <td>0.031339</td>
      <td>0.042841</td>
    </tr>
    <tr>
      <th>MITx/6.00x/2013_Spring</th>
      <td>0.944850</td>
      <td>0.046678</td>
      <td>0.021710</td>
      <td>0.046695</td>
    </tr>
    <tr>
      <th>MITx/8.02x/2013_Spring</th>
      <td>0.669447</td>
      <td>0.058039</td>
      <td>0.026475</td>
      <td>0.058071</td>
    </tr>
    <tr>
      <th>MITx/3.091x/2012_Fall</th>
      <td>0.493352</td>
      <td>0.063032</td>
      <td>0.044460</td>
      <td>0.063032</td>
    </tr>
    <tr>
      <th>MITx/6.00x/2012_Fall</th>
      <td>0.620716</td>
      <td>0.062685</td>
      <td>0.037119</td>
      <td>0.063074</td>
    </tr>
    <tr>
      <th>HarvardX/CS50x/2012</th>
      <td>0.625430</td>
      <td>0.064986</td>
      <td>0.007588</td>
      <td>0.065016</td>
    </tr>
    <tr>
      <th>HarvardX/ER22x/2013_Spring</th>
      <td>0.560238</td>
      <td>0.061527</td>
      <td>0.040867</td>
      <td>0.069435</td>
    </tr>
    <tr>
      <th>MITx/6.002x/2012_Fall</th>
      <td>0.637549</td>
      <td>0.073951</td>
      <td>0.042881</td>
      <td>0.074343</td>
    </tr>
    <tr>
      <th>MITx/7.00x/2013_Spring</th>
      <td>0.622686</td>
      <td>0.073826</td>
      <td>0.039174</td>
      <td>0.074825</td>
    </tr>
    <tr>
      <th>MITx/2.01x/2013_Spring</th>
      <td>0.682436</td>
      <td>0.098853</td>
      <td>0.043601</td>
      <td>0.099029</td>
    </tr>
    <tr>
      <th>HarvardX/PH207x/2012_Fall</th>
      <td>0.583742</td>
      <td>0.104155</td>
      <td>0.044287</td>
      <td>0.104179</td>
    </tr>
    <tr>
      <th>MITx/14.73x/2013_Spring</th>
      <td>0.588016</td>
      <td>0.105310</td>
      <td>0.074812</td>
      <td>0.105633</td>
    </tr>
  </tbody>
</table>
</div>



The goal is to attempt to predict engagement for a user based on data they do not have before they start the course.   This is only self-reported information like the Highest Level of Education, Year of Birth, Gender, and when the course started.   We can also attemp to predict it including the course id, but this will not generalize to other courses.  


    m1 = mooc[[u'LoE_DI', u'YoB', u'gender', u'start_time_DI',u'engagement']]


    for x in m1.LoE_DI.unique()[1:]:
        m1[x] = np.nan
        m1.loc[:,x] = np.where(m1.LoE_DI==x,1,0)
    
    m1.head()

    /Library/Python/2.7/site-packages/IPython/kernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from IPython.kernel.zmq import kernelapp as app





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoE_DI</th>
      <th>YoB</th>
      <th>gender</th>
      <th>start_time_DI</th>
      <th>engagement</th>
      <th>Secondary</th>
      <th>Bachelor's</th>
      <th>Master's</th>
      <th>Doctorate</th>
      <th>Less than Secondary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-12-19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-10-15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013-02-08</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-09-17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-12-19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    for x in m1.gender.unique()[1:]:
        m1[x] = np.nan
        m1.loc[:,x] = np.where(m1.gender==x,1,0)
    m1.head()


    /Library/Python/2.7/site-packages/IPython/kernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from IPython.kernel.zmq import kernelapp as app





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoE_DI</th>
      <th>YoB</th>
      <th>gender</th>
      <th>start_time_DI</th>
      <th>engagement</th>
      <th>Secondary</th>
      <th>Bachelor's</th>
      <th>Master's</th>
      <th>Doctorate</th>
      <th>Less than Secondary</th>
      <th>m</th>
      <th>f</th>
      <th>o</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-12-19</td>
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
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-10-15</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013-02-08</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-09-17</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-12-19</td>
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
</div>




    m1 = m1.drop(['LoE_DI','gender'],axis=1)
    m1.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YoB</th>
      <th>start_time_DI</th>
      <th>engagement</th>
      <th>Secondary</th>
      <th>Bachelor's</th>
      <th>Master's</th>
      <th>Doctorate</th>
      <th>Less than Secondary</th>
      <th>m</th>
      <th>f</th>
      <th>o</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>2012-12-19</td>
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
      <th>1</th>
      <td>NaN</td>
      <td>2012-10-15</td>
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
      <td>NaN</td>
      <td>2013-02-08</td>
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
      <td>NaN</td>
      <td>2012-09-17</td>
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
      <td>NaN</td>
      <td>2012-12-19</td>
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
</div>




    m1.YoB = pd.cut(m1.YoB,[0,1960,1970,1980,1990,2000,2010,2020])
    for x in m1.YoB.unique()[1:]:
        m1[x] = np.nan
        m1.loc[:,x] = np.where(m1.YoB==x,1,0)
    m1.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YoB</th>
      <th>start_time_DI</th>
      <th>engagement</th>
      <th>Secondary</th>
      <th>Bachelor's</th>
      <th>Master's</th>
      <th>Doctorate</th>
      <th>Less than Secondary</th>
      <th>m</th>
      <th>f</th>
      <th>o</th>
      <th>(2010, 2020]</th>
      <th>(1980, 1990]</th>
      <th>(1960, 1970]</th>
      <th>(1970, 1980]</th>
      <th>(1990, 2000]</th>
      <th>(0, 1960]</th>
      <th>(2000, 2010]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>2012-12-19</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>2012-10-15</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2013-02-08</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>2012-09-17</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>2012-12-19</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    m1 = m1.drop('YoB',axis=1)
    m1.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start_time_DI</th>
      <th>engagement</th>
      <th>Secondary</th>
      <th>Bachelor's</th>
      <th>Master's</th>
      <th>Doctorate</th>
      <th>Less than Secondary</th>
      <th>m</th>
      <th>f</th>
      <th>o</th>
      <th>(2010, 2020]</th>
      <th>(1980, 1990]</th>
      <th>(1960, 1970]</th>
      <th>(1970, 1980]</th>
      <th>(1990, 2000]</th>
      <th>(0, 1960]</th>
      <th>(2000, 2010]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-12-19</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-10-15</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-02-08</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-09-17</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-12-19</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    m1 = m1.drop(['start_time_DI'],axis=1)
    m1.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>engagement</th>
      <th>Secondary</th>
      <th>Bachelor's</th>
      <th>Master's</th>
      <th>Doctorate</th>
      <th>Less than Secondary</th>
      <th>m</th>
      <th>f</th>
      <th>o</th>
      <th>(2010, 2020]</th>
      <th>(1980, 1990]</th>
      <th>(1960, 1970]</th>
      <th>(1970, 1980]</th>
      <th>(1990, 2000]</th>
      <th>(0, 1960]</th>
      <th>(2000, 2010]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    m1.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 641138 entries, 0 to 641137
    Data columns (total 16 columns):
    engagement             641138 non-null int64
    Secondary              641138 non-null int64
    Bachelor's             641138 non-null int64
    Master's               641138 non-null int64
    Doctorate              641138 non-null int64
    Less than Secondary    641138 non-null int64
    m                      641138 non-null int64
    f                      641138 non-null int64
    o                      641138 non-null int64
    (2010, 2020]           641138 non-null int64
    (1980, 1990]           641138 non-null int64
    (1960, 1970]           641138 non-null int64
    (1970, 1980]           641138 non-null int64
    (1990, 2000]           641138 non-null int64
    (0, 1960]              641138 non-null int64
    (2000, 2010]           641138 non-null int64
    dtypes: int64(16)
    memory usage: 83.2 MB



    y = m1.engagement.values
    x = m1.drop('engagement',axis=1).values
    print y.shape,x.shape

    (641138,) (641138, 15)



    from sklearn.metrics import confusion_matrix
    
    a_train, a_test, b_train, b_test = train_test_split(x,y,test_size=0.50)
    
    mlog = LogisticRegressionCV(cv=20)
    mlog.fit(a_train,b_train)
    
    probs = mlog.predict_proba(a_test)[:,1]
    tpr1, fpr1, thresholds1 = roc_curve(probs, b_test)
    
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("MOOC")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc=4)
    plt.show()
    print "Recall: ", precision_score(mlog.predict(a_test),b_test)
    print "Precision: ", recall_score(mlog.predict(a_test),b_test)
    print "Accuracy: ", accuracy_score(mlog.predict(a_test),b_test)
    confusion_matrix(mlog.predict(a_test),b_test)



![png](http://www.bryantravissmith.com/img/GW03D4/output_60_0.png)


    Recall:  0.0
    Precision:  0.0
    Accuracy:  0.937143017572





    array([[300419,  20150],
           [     0,      0]])



This model, unsurpisingly, predicts no engagement because engagement is so rare.  This is a know problem with logistic regresion.   To avoid this we can drop duplicates and refit the data.


    m2 = m1.drop_duplicates()
    y = m2.engagement.values
    x = m2.drop('engagement',axis=1).values
    print y.shape,x.shape

    (163,) (163, 15)



    a_train, a_test, b_train, b_test = train_test_split(x,y,test_size=0.50)
    
    mlog = LogisticRegressionCV(cv=20)
    mlog.fit(a_train,b_train)
    
    probs = mlog.predict_proba(a_test)[:,1]
    tpr1, fpr1, thresholds1 = roc_curve(probs, b_test)
    
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("MOOC")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc=4)
    plt.show()
    print "Recall: ", precision_score(mlog.predict(a_test),b_test)
    print "Precision: ", recall_score(mlog.predict(a_test),b_test)
    print "Accuracy: ", accuracy_score(mlog.predict(a_test),b_test)
    confusion_matrix(mlog.predict(a_test),b_test)


![png](http://www.bryantravissmith.com/img/GW03D4/output_63_0.png)


    Recall:  0.121951219512
    Precision:  0.277777777778
    Accuracy:  0.40243902439





    array([[28, 36],
           [13,  5]])




    probs = mlog.predict_proba(m1.drop('engagement',axis=1).values)[:,1]
    predict = mlog.predict(m1.drop('engagement',axis=1).values)
    tpr1, fpr1, thresholds1 = roc_curve(probs, m1.engagement.values)
    
    plt.plot(fpr, tpr,color='seagreen',alpha=0.8,lw=3)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("MOOC")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc=4)
    plt.show()
    print "Recall: ", precision_score(predict,m1.engagement.values)
    print "Precision: ", recall_score(predict,m1.engagement.values)
    print "Accuracy: ", accuracy_score(predict,m1.engagement.values)
    confusion_matrix(predict,m1.engagement.values)


![png](http://www.bryantravissmith.com/img/GW03D4/output_64_0.png)


    Recall:  0.136417673866
    Precision:  0.0649505324104
    Accuracy:  0.821936930895





    array([[521467,  34868],
           [ 79295,   5508]])



I do not think the model is any better.   Ad at this point I think based on this data we are not able to predict engagement by self-reported features


    
