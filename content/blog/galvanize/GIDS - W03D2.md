Title: Galvanize - Week 03 - Day 2
Date: 2015-06-16 10:20
Modified: 2015-06-16 10:30
Category: Galvanize
Tags: data-science, galvanize, regression
Slug: galvanize-data-science-03-02
Authors: Bryan Smith
Summary: The twelvth day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered linear regression.

#Galvanize Immersive Data Science
##Week 3 - Day 2

Today we did more linear regression, and started evaluating applying linear models to regression problems.   It was a combination of informative and frustration because I learned a lot, but did not develope an intuition for the process.   I suppsoe that will come with time.  

##Mini-Quiz

Today's miniquiz took all of 5 minutes.

1. Write a python function to find the value in a list that's closest to a given value.

    e.g. `closest([10, 17, 2, 29, 16], 14` should return 16.


    a = [10,17,2,29,16]
    import math
    def closest(array,val):
        diff = [math.fabs(x-val) for x in array]
        index = diff.index(min(diff))
        return array[index]
    
    closest(a,14)




    16



2. Instead let's start with a numpy array. How can we do the same thing in one line using numpy magic.


    import numpy as np
    
    a = np.array(a)
    a[np.abs(a-14).argmin()]




    16



3. My favorite numpy trick is [masking](http://docs.scipy.org/doc/numpy/user/basics.indexing.html#boolean-or-mask-index-arrays). Say you have a feature matrix `X` (2d numpy array) and with labels `y` (1d numpy array). I would like to get a feature matrix of only the positive cases, i.e. get the rows from `X` where `y` is positive.

    How can you do this in one line?
    
    Create example `X` and `y` to verify your code.


    x = np.random.rand(100,10)
    y = np.random.randint(2,size=(100,))
    
    print x.shape, x[y>0,:].shape

    (100, 10) (47, 10)



    print set(y[y>0].tolist())

    set([1])


##Morning - Regression
The linear regression model makes a number of assumptions about the data, including 

- **Homoscedasticity of residuals**
- **Normal distribution of residuals**
- **Lack of multicollinearity among features**
- **Independence of the observations (For example, independence assumption violated if data is a time series)**

Since the results of the regression model depend on these statistical assumptions, the 
results of the regression model are only correct if our assumptions hold (at least approximately).

<br>

This morning we will be exploring two datasets: `prestige` and `ccard`. Below is a description of the 2 datasets.

* `prestige` _(From yesterday afternoon)_
    - Prediction of the prestige of a job
    - Dependent variable: `prestige`
    - Independent variables: `income`, `education`
    - Code to load data set into dataframe:
  
  ```python
  import statsmodels.api as sm
  prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
  ```
   
* `ccard`
    - Prediction of the average credit card expenditure
    - Dependent variable: `AVGEXP`
    - Independent variables: `AGE`, `INCOME`, `INCOMESQ` (`INCOME^2`), `OWNRENT`
    - Code to load data set into dataframe:
  
  ```python
  credit_card = sm.datasets.ccard.load_pandas().data
  ```

##Prestige Regression


    %matplotlib inline
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import scipy.stats as sc
    import statsmodels.api as sm
    from pandas.tools.plotting import scatter_matrix
    prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
    credit_card = sm.datasets.ccard.load_pandas().data


    plt.figure(figsize=(10,10))
    scatter_matrix(prestige,diagonal='kde',figsize=(14,14),alpha=0.4)
    plt.show()


    <matplotlib.figure.Figure at 0x106186450>



![png](http://www.bryantravissmith.com/img/output_9_1.png)



    plt.figure()
    prestige.boxplot()
    plt.show()

    /Library/Python/2.7/site-packages/pandas/tools/plotting.py:2633: FutureWarning: 
    The default value for 'return_type' will change to 'axes' in a future release.
     To use the future behavior now, set return_type='axes'.
     To keep the previous behavior and silence this warning, set return_type='dict'.
      warnings.warn(msg, FutureWarning)



![png](http://www.bryantravissmith.com/img/output_10_1.png)



    y = prestige[['prestige']]
    x = sm.add_constant(prestige[['income','education']])
    model1 = sm.OLS(y,x).fit()
    model1.summary()




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
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>8.65e-17</td>
</tr>
<tr>
  <th>Time:</th>                 <td>07:50:23</td>     <th>  Log-Likelihood:    </th> <td> -178.98</td>
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



So far we have looked at the data distributions, and the scatter matrix shows relationships between prestige and both income and education.  We have some covariance between income and educations, that could affect the assumption of linear regression that the variables in the model are independant.   We can look at a studentized resid plot and a QQ plot to examine the model.


    def stud_plot(model):
        stud = model.resid/model.resid.std()
        fitv = model.fittedvalues
        plt.figure(figsize=(10,4))
        plt.plot(fitv,stud,marker='o',lw=0,color='steelblue',alpha=0.8)
        plt.axhline(2,color='gray',linestyle='--',)
        plt.axhline(-2,color='gray',linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Studentized Residual")
        plt.ylim([-5,5])
        plt.show()
        
    def qq_plot(model):
        plt.figure(figsize=(10,4))
        stud = model.resid/model.resid.std()
        sc.probplot(model.resid,plot=plt)
        plt.title("QQ Plot")
        plt.show()
        
    stud_plot(model1)


![png](http://www.bryantravissmith.com/img/output_13_0.png)



    qq_plot(model1)


![png](http://www.bryantravissmith.com/img/output_14_0.png)


We see se have two outliers from the studentized plot, those that are more than 2 standard deviation from the fit.   The QQ-plot shows a fairly normal distributions of the residuals, suggesting the fit is okay.  We can also see if we have any points with high leverage.

$$\hat{y} = X \dot \beta = X (X^T \ X)^{-1} \ X^T \ y = H \ y$$

$$ H = X (X^T \ X)^{-1} \ X^T $$  

The diagonals of the hat matrix (H) are considered to be estimates of leverage of each point.


    def get_H(X):
        H = X.dot( np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()) )
        return H.diagonal()
    
    def lev_plot(model,X):
        h = get_H(X)
        stud = model.resid/model.resid.std()
        plt.figure(figsize=(10,5))
        plt.plot(h,stud,marker='o',lw=0,color='steelblue',alpha=0.8)
        plt.xlabel("H Leverage")
        plt.ylabel("Studentized Residual")
        plt.ylim([-5,5])
        plt.show()
        
    lev_plot(model1,x.values)


![png](http://www.bryantravissmith.com/img/output_17_0.png)


We can see that we have 3 points that are leveraging the results, and one of them is an outlier.

Now we have the question - what do we do with the outliers and the high leverage points?  I think we will stick wih removing all the high leverage points.  We before we do lets look that these three points.


    x[get_H(x.values)>0.15]




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
      <th>minister</th>
      <td>1</td>
      <td>21</td>
      <td>84</td>
    </tr>
    <tr>
      <th>conductor</th>
      <td>1</td>
      <td>76</td>
      <td>34</td>
    </tr>
    <tr>
      <th>RR.engineer</th>
      <td>1</td>
      <td>81</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




    y[get_H(x.values)>=0.15]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prestige</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>minister</th>
      <td>87</td>
    </tr>
    <tr>
      <th>conductor</th>
      <td>38</td>
    </tr>
    <tr>
      <th>RR.engineer</th>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>



We see the engineer and conductor have higher income, but lower eduation, while the minister has lower education and higher income.   To decide what we do with these values we really need to know the purpose of the model.  


    hp = get_H(x.values)
    model2 = sm.OLS(y[hp<0.15],x[hp<0.15]).fit()
    model2.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>prestige</td>     <th>  R-squared:         </th> <td>   0.876</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.870</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   138.1</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>2.02e-18</td>
</tr>
<tr>
  <th>Time:</th>                 <td>07:50:23</td>     <th>  Log-Likelihood:    </th> <td> -160.59</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    42</td>      <th>  AIC:               </th> <td>   327.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    39</td>      <th>  BIC:               </th> <td>   332.4</td>
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
  <th>const</th>     <td>   -6.3174</td> <td>    3.680</td> <td>   -1.717</td> <td> 0.094</td> <td>  -13.760     1.125</td>
</tr>
<tr>
  <th>income</th>    <td>    0.9307</td> <td>    0.154</td> <td>    6.053</td> <td> 0.000</td> <td>    0.620     1.242</td>
</tr>
<tr>
  <th>education</th> <td>    0.2846</td> <td>    0.121</td> <td>    2.345</td> <td> 0.024</td> <td>    0.039     0.530</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.811</td> <th>  Durbin-Watson:     </th> <td>   1.468</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.149</td> <th>  Jarque-Bera (JB):  </th> <td>   2.802</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.614</td> <th>  Prob(JB):          </th> <td>   0.246</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.303</td> <th>  Cond. No.          </th> <td>    158.</td>
</tr>
</table>



Removing 3 of the 45 data points slightly improved the R-squared value, but dramatically changed the model.   The coefficients were initialy (0.5987,0.5458).  The new model coefficient is (0.9407,2846).  The minister must have been pulling the fit down because they have high prestigue but low income.  Removing them changed the model.   Few people become ministers, so it could be a better model without including them. 


    stud_plot(model2)
    qq_plot(model2)
    lev_plot(model2,x[get_H(x.values)<0.15].values)


![png](http://www.bryantravissmith.com/img/output_24_0.png)



![png](http://www.bryantravissmith.com/img/output_24_1.png)



![png](http://www.bryantravissmith.com/img/output_24_2.png)


The leverage plot does not show a few points dominating the fit, buter there are a few outliers in the fit.   The QQ plot shows some signs of the erros not being normal.   


    print "Model 1 RSE: ", np.sqrt(np.sum(np.power(y.values.reshape(1,45)-model1.predict(x),2))/(45-2-1))
    print "Model 2 RSE: ", np.sqrt(np.sum(np.power(y.values.reshape(1,45)-model2.predict(x),2))/(45-2-1))

    Model 1 RSE:  13.3690283982
    Model 2 RSE:  14.6713753944


On the entire dataset, the first model does a better job fitting the entire dataset then the second model.  The second model fits the subsetted data better.   On these vlaues we have


    print "Model 1 RSE: ", np.sqrt(np.sum(np.power(y[hp<0.15].values.reshape(1,42)-model1.predict(x[hp<0.15]),2))/(42-2-1))
    print "Model 2 RSE: ", np.sqrt(np.sum(np.power(y[hp<0.15].values.reshape(1,42)-model2.predict(x[hp<0.15]),2))/(42-2-1))

    Model 1 RSE:  12.2166311202
    Model 2 RSE:  11.492268427


On the subsetted data, the second model does do better.   If this model was unleashed into the wild we would need to identify outliers and attempt to decide what to do with them

##Credit Analysis


    plt.figure(figsize=(10,10))
    scatter_matrix(credit_card,diagonal='kde',alpha=0.5,figsize=(14,14))
    plt.show()


    <matplotlib.figure.Figure at 0x10a815250>



![png](http://www.bryantravissmith.com/img/output_30_1.png)



    credit_card.corr()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVGEXP</th>
      <th>AGE</th>
      <th>INCOME</th>
      <th>INCOMESQ</th>
      <th>OWNRENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVGEXP</th>
      <td>1.000000</td>
      <td>0.168367</td>
      <td>0.443135</td>
      <td>0.372679</td>
      <td>0.243342</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.168367</td>
      <td>1.000000</td>
      <td>0.385108</td>
      <td>0.316498</td>
      <td>0.438236</td>
    </tr>
    <tr>
      <th>INCOME</th>
      <td>0.443135</td>
      <td>0.385108</td>
      <td>1.000000</td>
      <td>0.964707</td>
      <td>0.473079</td>
    </tr>
    <tr>
      <th>INCOMESQ</th>
      <td>0.372679</td>
      <td>0.316498</td>
      <td>0.964707</td>
      <td>1.000000</td>
      <td>0.434500</td>
    </tr>
    <tr>
      <th>OWNRENT</th>
      <td>0.243342</td>
      <td>0.438236</td>
      <td>0.473079</td>
      <td>0.434500</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




    credit_card.corr(method='spearman')




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AVGEXP</th>
      <th>AGE</th>
      <th>INCOME</th>
      <th>INCOMESQ</th>
      <th>OWNRENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AVGEXP</th>
      <td>1.000000</td>
      <td>0.182947</td>
      <td>0.533556</td>
      <td>0.533556</td>
      <td>0.340274</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.182947</td>
      <td>1.000000</td>
      <td>0.444940</td>
      <td>0.444940</td>
      <td>0.434660</td>
    </tr>
    <tr>
      <th>INCOME</th>
      <td>0.533556</td>
      <td>0.444940</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.410097</td>
    </tr>
    <tr>
      <th>INCOMESQ</th>
      <td>0.533556</td>
      <td>0.444940</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.410097</td>
    </tr>
    <tr>
      <th>OWNRENT</th>
      <td>0.340274</td>
      <td>0.434660</td>
      <td>0.410097</td>
      <td>0.410097</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The scatter matrix shows that therre are some variations in between the average expense and age, income, and income squaree.  Obviously income and income squared are related.   The spearmann correlations show that age and income are ordered together, but that ordering does not seem to be a linear relationship because of the significant increason from pearson to spearman.  This could affect our modeling.   

We'll do a basic fit and see what we are starting with.


    yc = credit_card[['AVGEXP']]
    xc = sm.add_constant(credit_card[['INCOME','INCOMESQ','AGE',"OWNRENT"]])
    model3 = sm.OLS(yc,xc).fit()
    model3.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>AVGEXP</td>      <th>  R-squared:         </th> <td>   0.244</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.198</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   5.394</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>0.000795</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:06:50</td>     <th>  Log-Likelihood:    </th> <td> -506.49</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    72</td>      <th>  AIC:               </th> <td>   1023.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    67</td>      <th>  BIC:               </th> <td>   1034.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td> -237.1465</td> <td>  199.352</td> <td>   -1.190</td> <td> 0.238</td> <td> -635.054   160.761</td>
</tr>
<tr>
  <th>INCOME</th>   <td>  234.3470</td> <td>   80.366</td> <td>    2.916</td> <td> 0.005</td> <td>   73.936   394.758</td>
</tr>
<tr>
  <th>INCOMESQ</th> <td>  -14.9968</td> <td>    7.469</td> <td>   -2.008</td> <td> 0.049</td> <td>  -29.906    -0.088</td>
</tr>
<tr>
  <th>AGE</th>      <td>   -3.0818</td> <td>    5.515</td> <td>   -0.559</td> <td> 0.578</td> <td>  -14.089     7.926</td>
</tr>
<tr>
  <th>OWNRENT</th>  <td>   27.9409</td> <td>   82.922</td> <td>    0.337</td> <td> 0.737</td> <td> -137.573   193.455</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>69.024</td> <th>  Durbin-Watson:     </th> <td>   1.640</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 497.349</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 2.844</td> <th>  Prob(JB):          </th> <td>1.00e-108</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>14.551</td> <th>  Cond. No.          </th> <td>    227.</td> 
</tr>
</table>




    stud_plot(model3)
    qq_plot(model3)
    lev_plot(model3,xc.values)


![png](http://www.bryantravissmith.com/img/output_35_0.png)



![png](http://www.bryantravissmith.com/img/output_35_1.png)



![png](http://www.bryantravissmith.com/img/output_35_2.png)


Our inital linear model is not a strong fit, age and own/rent are not significantly involved in the model, the residuals are not normally distributed, and there are outliers and high leverage points.   We clearly need to do something with this model.  The variances does not look constant.


    from statsmodels.stats.diagnostic import HetGoldfeldQuandt
    HGQ = HetGoldfeldQuandt()
    HGQ.run(yc,xc)




    (1.3799238831599079, 0.18743313399026834, 'increasing')




    def equal_var(model):
        test = np.ones((len(model.fittedvalues),2))
        test[:,0] = model.fittedvalues
        test[:,1] = model.resid/model.resid.std()
        test = test[test[:,0].argsort()]
        half = len(model.fittedvalues)/2
        return sc.levene(test[:half,1],test[half:,1])
    
    equal_var(model3)




    (7.6983153917314482, 0.0070815962364431974)



We see that th HetGoldfeldQuandt test shows the variance of the data is not significantly increasing with fitted values, but the levene test shows that there is not equal variance between the lower half and the upper half of the results.   If true, we need to have a fundamentally different model.  We could try a power law relationship.


    
    model4 = sm.OLS(np.log(yc+1),xc).fit()
    model4.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>AVGEXP</td>      <th>  R-squared:         </th> <td>   0.282</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.239</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   6.576</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>0.000159</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:01:19</td>     <th>  Log-Likelihood:    </th> <td> -97.129</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    72</td>      <th>  AIC:               </th> <td>   204.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    67</td>      <th>  BIC:               </th> <td>   215.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td>    3.6758</td> <td>    0.677</td> <td>    5.432</td> <td> 0.000</td> <td>    2.325     5.027</td>
</tr>
<tr>
  <th>INCOME</th>   <td>    0.7680</td> <td>    0.273</td> <td>    2.815</td> <td> 0.006</td> <td>    0.223     1.313</td>
</tr>
<tr>
  <th>INCOMESQ</th> <td>   -0.0463</td> <td>    0.025</td> <td>   -1.826</td> <td> 0.072</td> <td>   -0.097     0.004</td>
</tr>
<tr>
  <th>AGE</th>      <td>   -0.0238</td> <td>    0.019</td> <td>   -1.273</td> <td> 0.207</td> <td>   -0.061     0.014</td>
</tr>
<tr>
  <th>OWNRENT</th>  <td>    0.3307</td> <td>    0.281</td> <td>    1.175</td> <td> 0.244</td> <td>   -0.231     0.893</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.638</td> <th>  Durbin-Watson:     </th> <td>   1.902</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.060</td> <th>  Jarque-Bera (JB):  </th> <td>   4.883</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.511</td> <th>  Prob(JB):          </th> <td>  0.0870</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.764</td> <th>  Cond. No.          </th> <td>    227.</td>
</tr>
</table>




    stud_plot(model4)
    qq_plot(model4)
    lev_plot(model4,xc.values)


![png](http://www.bryantravissmith.com/img/output_41_0.png)



![png](http://www.bryantravissmith.com/img/output_41_1.png)



![png](http://www.bryantravissmith.com/img/output_41_2.png)



    HGQ = HetGoldfeldQuandt()
    print HGQ.run(np.log(yc+1),xc)
    print equal_var(model4)

    (0.54047445064175459, 0.95411861481885185, 'increasing')
    (0.0049114588257328637, 0.94432837966256522)


Both tests show that the variance appears to be constant with this fit.   The exponential model seems to be an improvement from this perspective.   We can look at the variance indication factor to see if there are any variables we should not be including.


    def print_vif(x):
        for i,col in enumerate(x.columns):
            print col, vif(x.values,i)
    print_vif(xc)

    const 35.2892427253
    INCOME 16.3339165879
    INCOMESQ 15.21011121
    AGE 1.3624340126
    OWNRENT 1.43105661679


In this case the income and income square obviously predict each other, but the other variables appear to be independant.   Maybe this is the case for an interaction term in our model.  Owners and renters probably have different credit habits.  We also saw spearman correlation between age and income.  The improvement of this model will have to be explored after we cover interaction terms

##Afternoon

In the afternoon we were given a new data set and told to perform a regression fit on the data.


    balance = pd.read_csv('../linear-regression/data/balance.csv')
    balance.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 400 entries, 0 to 399
    Data columns (total 12 columns):
    Unnamed: 0    400 non-null int64
    Income        400 non-null float64
    Limit         400 non-null int64
    Rating        400 non-null int64
    Cards         400 non-null int64
    Age           400 non-null int64
    Education     400 non-null int64
    Gender        400 non-null object
    Student       400 non-null object
    Married       400 non-null object
    Ethnicity     400 non-null object
    Balance       400 non-null int64
    dtypes: float64(1), int64(7), object(4)
    memory usage: 40.6+ KB



    balance.head(10)




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>80.180</td>
      <td>8047</td>
      <td>569</td>
      <td>4</td>
      <td>77</td>
      <td>10</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Caucasian</td>
      <td>1151</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>20.996</td>
      <td>3388</td>
      <td>259</td>
      <td>2</td>
      <td>37</td>
      <td>12</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>African American</td>
      <td>203</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>71.408</td>
      <td>7114</td>
      <td>512</td>
      <td>2</td>
      <td>87</td>
      <td>9</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>872</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>15.125</td>
      <td>3300</td>
      <td>266</td>
      <td>5</td>
      <td>66</td>
      <td>13</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Caucasian</td>
      <td>279</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>71.061</td>
      <td>6819</td>
      <td>491</td>
      <td>3</td>
      <td>41</td>
      <td>19</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>African American</td>
      <td>1350</td>
    </tr>
  </tbody>
</table>
</div>




    scatter_matrix(balance,diagonal='kde',figsize=(14,14))
    plt.show()


![png](http://www.bryantravissmith.com/img/output_48_0.png)


In order to do regression we need to remove the categorical variables and replace them with variables that are zero or one.  With Ethnicity we need to decided ona  baseline, which we decided on african american (alphabetical order).


    balance = balance.drop("Unnamed: 0",axis=1)
    balance.Gender = balance.Gender.str.strip()
    balance.Gender = np.where(balance.Gender=='Male',1,0)
    balance.Student = np.where(balance.Student=='Yes',1,0)
    balance.Married = np.where(balance.Married=='Yes',1,0)
    balance['Caucasian'] = np.where(balance.Ethnicity=='Caucasian',1,0)
    balance['Asian'] = np.where(balance.Ethnicity=='Asian',1,0)
    balance = balance.drop('Ethnicity',axis=1)
    balance.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Balance</th>
      <th>Caucasian</th>
      <th>Asian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>333</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>903</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>580</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>964</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>331</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    yb = balance[['Balance']]
    xb = sm.add_constant(balance.drop(['Balance'],axis=1))
    modelb1 = sm.OLS(yb,xb).fit()
    modelb1.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.955</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.954</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   750.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>1.11e-253</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:33:39</td>     <th>  Log-Likelihood:    </th> <td> -2398.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   4821.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   388</td>      <th>  BIC:               </th> <td>   4869.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
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
  <th>const</th>     <td> -489.8611</td> <td>   35.801</td> <td>  -13.683</td> <td> 0.000</td> <td> -560.250  -419.473</td>
</tr>
<tr>
  <th>Income</th>    <td>   -7.8031</td> <td>    0.234</td> <td>  -33.314</td> <td> 0.000</td> <td>   -8.264    -7.343</td>
</tr>
<tr>
  <th>Limit</th>     <td>    0.1909</td> <td>    0.033</td> <td>    5.824</td> <td> 0.000</td> <td>    0.126     0.255</td>
</tr>
<tr>
  <th>Rating</th>    <td>    1.1365</td> <td>    0.491</td> <td>    2.315</td> <td> 0.021</td> <td>    0.171     2.102</td>
</tr>
<tr>
  <th>Cards</th>     <td>   17.7245</td> <td>    4.341</td> <td>    4.083</td> <td> 0.000</td> <td>    9.190    26.259</td>
</tr>
<tr>
  <th>Age</th>       <td>   -0.6139</td> <td>    0.294</td> <td>   -2.088</td> <td> 0.037</td> <td>   -1.192    -0.036</td>
</tr>
<tr>
  <th>Education</th> <td>   -1.0989</td> <td>    1.598</td> <td>   -0.688</td> <td> 0.492</td> <td>   -4.241     2.043</td>
</tr>
<tr>
  <th>Gender</th>    <td>   10.6532</td> <td>    9.914</td> <td>    1.075</td> <td> 0.283</td> <td>   -8.839    30.145</td>
</tr>
<tr>
  <th>Student</th>   <td>  425.7474</td> <td>   16.723</td> <td>   25.459</td> <td> 0.000</td> <td>  392.869   458.626</td>
</tr>
<tr>
  <th>Married</th>   <td>   -8.5339</td> <td>   10.363</td> <td>   -0.824</td> <td> 0.411</td> <td>  -28.908    11.841</td>
</tr>
<tr>
  <th>Caucasian</th> <td>   10.1070</td> <td>   12.210</td> <td>    0.828</td> <td> 0.408</td> <td>  -13.899    34.113</td>
</tr>
<tr>
  <th>Asian</th>     <td>   16.8042</td> <td>   14.119</td> <td>    1.190</td> <td> 0.235</td> <td>  -10.955    44.564</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>34.899</td> <th>  Durbin-Watson:     </th> <td>   1.968</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  41.766</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.782</td> <th>  Prob(JB):          </th> <td>8.52e-10</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.241</td> <th>  Cond. No.          </th> <td>3.87e+04</td>
</tr>
</table>




    stud_plot(modelb1)
    qq_plot(modelb1)
    lev_plot(modelb1,xb.values)


![png](http://www.bryantravissmith.com/img/output_52_0.png)



![png](http://www.bryantravissmith.com/img/output_52_1.png)



![png](http://www.bryantravissmith.com/img/output_52_2.png)


We have some very strange behavor with our model.   I am currious what is happening here.  Lets start by looking at the data points where we have the strang line on the studentized plot.


    balance[modelb1.fittedvalues < 100].describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Balance</th>
      <th>Caucasian</th>
      <th>Asian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81</td>
      <td>81.000000</td>
      <td>81.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.950296</td>
      <td>2078.024691</td>
      <td>177.074074</td>
      <td>2.753086</td>
      <td>55.679012</td>
      <td>13.419753</td>
      <td>0.530864</td>
      <td>0.012346</td>
      <td>0.592593</td>
      <td>0</td>
      <td>0.481481</td>
      <td>0.308642</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.648895</td>
      <td>736.739048</td>
      <td>47.463085</td>
      <td>1.145981</td>
      <td>17.237914</td>
      <td>2.801179</td>
      <td>0.502156</td>
      <td>0.111111</td>
      <td>0.494413</td>
      <td>0</td>
      <td>0.502770</td>
      <td>0.464811</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.726000</td>
      <td>855.000000</td>
      <td>93.000000</td>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.036000</td>
      <td>1501.000000</td>
      <td>142.000000</td>
      <td>2.000000</td>
      <td>40.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.039000</td>
      <td>2047.000000</td>
      <td>173.000000</td>
      <td>3.000000</td>
      <td>57.000000</td>
      <td>14.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>34.772000</td>
      <td>2529.000000</td>
      <td>199.000000</td>
      <td>3.000000</td>
      <td>70.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>92.112000</td>
      <td>4612.000000</td>
      <td>344.000000</td>
      <td>6.000000</td>
      <td>84.000000</td>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




    balance[modelb1.fittedvalues > 100].describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Balance</th>
      <th>Caucasian</th>
      <th>Asian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
      <td>319.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>49.349781</td>
      <td>5410.407524</td>
      <td>400.103448</td>
      <td>3.009404</td>
      <td>55.664577</td>
      <td>13.457680</td>
      <td>0.470219</td>
      <td>0.122257</td>
      <td>0.617555</td>
      <td>652.056426</td>
      <td>0.501567</td>
      <td>0.241379</td>
    </tr>
    <tr>
      <th>std</th>
      <td>37.582144</td>
      <td>2071.839874</td>
      <td>139.162472</td>
      <td>1.419730</td>
      <td>17.279892</td>
      <td>3.206312</td>
      <td>0.499896</td>
      <td>0.328097</td>
      <td>0.486748</td>
      <td>422.907364</td>
      <td>0.500783</td>
      <td>0.428592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.354000</td>
      <td>1160.000000</td>
      <td>126.000000</td>
      <td>1.000000</td>
      <td>23.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.022500</td>
      <td>3907.000000</td>
      <td>298.000000</td>
      <td>2.000000</td>
      <td>42.500000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>305.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.508000</td>
      <td>5110.000000</td>
      <td>377.000000</td>
      <td>3.000000</td>
      <td>56.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>607.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>62.848500</td>
      <td>6408.000000</td>
      <td>467.000000</td>
      <td>4.000000</td>
      <td>69.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>950.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>186.634000</td>
      <td>13913.000000</td>
      <td>982.000000</td>
      <td>9.000000</td>
      <td>98.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1999.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




    balance.describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Balance</th>
      <th>Caucasian</th>
      <th>Asian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.00000</td>
      <td>400.000000</td>
      <td>400.00000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>45.218885</td>
      <td>4735.600000</td>
      <td>354.940000</td>
      <td>2.957500</td>
      <td>55.667500</td>
      <td>13.450000</td>
      <td>0.482500</td>
      <td>0.100000</td>
      <td>0.61250</td>
      <td>520.015000</td>
      <td>0.49750</td>
      <td>0.255000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>35.244273</td>
      <td>2308.198848</td>
      <td>154.724143</td>
      <td>1.371275</td>
      <td>17.249807</td>
      <td>3.125207</td>
      <td>0.500319</td>
      <td>0.300376</td>
      <td>0.48779</td>
      <td>459.758877</td>
      <td>0.50062</td>
      <td>0.436407</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.354000</td>
      <td>855.000000</td>
      <td>93.000000</td>
      <td>1.000000</td>
      <td>23.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.007250</td>
      <td>3088.000000</td>
      <td>247.250000</td>
      <td>2.000000</td>
      <td>41.750000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>68.750000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.115500</td>
      <td>4622.500000</td>
      <td>344.000000</td>
      <td>3.000000</td>
      <td>56.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>459.500000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>57.470750</td>
      <td>5872.750000</td>
      <td>437.250000</td>
      <td>4.000000</td>
      <td>70.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>863.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>186.634000</td>
      <td>13913.000000</td>
      <td>982.000000</td>
      <td>9.000000</td>
      <td>98.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1999.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We see that relative to the averages, the trouble points looked to be people with no balance.  If we do a fit without balance, just to check, I am curreous if we get a good model.


    modelb2 = sm.OLS(yb[yb.values>0],xb[yb.values>0]).fit()
    modelb2.summary()





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.999</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.999</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>4.366e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>08:44:21</td>     <th>  Log-Likelihood:    </th> <td> -1162.5</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   310</td>      <th>  AIC:               </th> <td>   2349.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   298</td>      <th>  BIC:               </th> <td>   2394.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
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
  <th>const</th>     <td> -696.6745</td> <td>    4.406</td> <td> -158.103</td> <td> 0.000</td> <td> -705.346  -688.003</td>
</tr>
<tr>
  <th>Income</th>    <td>   -9.9916</td> <td>    0.029</td> <td> -339.458</td> <td> 0.000</td> <td>  -10.050    -9.934</td>
</tr>
<tr>
  <th>Limit</th>     <td>    0.3360</td> <td>    0.004</td> <td>   84.135</td> <td> 0.000</td> <td>    0.328     0.344</td>
</tr>
<tr>
  <th>Rating</th>    <td>   -0.1433</td> <td>    0.059</td> <td>   -2.428</td> <td> 0.016</td> <td>   -0.259    -0.027</td>
</tr>
<tr>
  <th>Cards</th>     <td>   25.4764</td> <td>    0.500</td> <td>   50.962</td> <td> 0.000</td> <td>   24.493    26.460</td>
</tr>
<tr>
  <th>Age</th>       <td>   -1.0029</td> <td>    0.036</td> <td>  -28.215</td> <td> 0.000</td> <td>   -1.073    -0.933</td>
</tr>
<tr>
  <th>Education</th> <td>   -0.0080</td> <td>    0.189</td> <td>   -0.042</td> <td> 0.966</td> <td>   -0.381     0.365</td>
</tr>
<tr>
  <th>Gender</th>    <td>   -0.2332</td> <td>    1.200</td> <td>   -0.194</td> <td> 0.846</td> <td>   -2.596     2.129</td>
</tr>
<tr>
  <th>Student</th>   <td>  500.8310</td> <td>    1.880</td> <td>  266.464</td> <td> 0.000</td> <td>  497.132   504.530</td>
</tr>
<tr>
  <th>Married</th>   <td>   -2.0625</td> <td>    1.261</td> <td>   -1.636</td> <td> 0.103</td> <td>   -4.543     0.418</td>
</tr>
<tr>
  <th>Caucasian</th> <td>   -0.0700</td> <td>    1.467</td> <td>   -0.048</td> <td> 0.962</td> <td>   -2.956     2.816</td>
</tr>
<tr>
  <th>Asian</th>     <td>   -1.3785</td> <td>    1.731</td> <td>   -0.796</td> <td> 0.426</td> <td>   -4.784     2.027</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.799</td> <th>  Durbin-Watson:     </th> <td>   2.067</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.150</td> <th>  Jarque-Bera (JB):  </th> <td>   3.766</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.269</td> <th>  Prob(JB):          </th> <td>   0.152</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.963</td> <th>  Cond. No.          </th> <td>4.37e+04</td>
</tr>
</table>




    stud_plot(modelb2)
    qq_plot(modelb2)
    lev_plot(modelb2,xb[yb.values>0].values)


![png](http://www.bryantravissmith.com/img/output_59_0.png)



![png](http://www.bryantravissmith.com/img/output_59_1.png)



![png](http://www.bryantravissmith.com/img/output_59_2.png)


So we have a great fit where the variance is constant, the residuals are normally distributed, and there are no high leverage points if we fit the data with removing the zero-balance points.   People with zero balance, however, are in the data.   Our regression does not make valid predictions for people with zero balnce.  

This seems like a two step process:

1.  Classify if the person is likely to have zero balance.
2.  If they do predict zero, otherwise predict the regression model.

Another solution would be to cut on credit rating.  One model for those above 350 credit rating, and another model for those with less than 350 credit rating.  This is because in the zero balance group, the max credit rating was 344.  Some of the issues is that we not know if this is a general cutoff, but lets try it.




    modelb3 = sm.OLS(yb[xb.Rating > 350],xb[xb.Rating > 350]).fit()
    modelb3.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.999</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.999</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.854e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>1.07e-273</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:06:41</td>     <th>  Log-Likelihood:    </th> <td> -741.54</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   196</td>      <th>  AIC:               </th> <td>   1507.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   184</td>      <th>  BIC:               </th> <td>   1546.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
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
  <th>const</th>     <td> -691.8364</td> <td>    6.766</td> <td> -102.258</td> <td> 0.000</td> <td> -705.184  -678.488</td>
</tr>
<tr>
  <th>Income</th>    <td>   -9.9829</td> <td>    0.037</td> <td> -271.846</td> <td> 0.000</td> <td>  -10.055    -9.910</td>
</tr>
<tr>
  <th>Limit</th>     <td>    0.3346</td> <td>    0.005</td> <td>   65.301</td> <td> 0.000</td> <td>    0.325     0.345</td>
</tr>
<tr>
  <th>Rating</th>    <td>   -0.1283</td> <td>    0.076</td> <td>   -1.698</td> <td> 0.091</td> <td>   -0.277     0.021</td>
</tr>
<tr>
  <th>Cards</th>     <td>   25.3670</td> <td>    0.660</td> <td>   38.407</td> <td> 0.000</td> <td>   24.064    26.670</td>
</tr>
<tr>
  <th>Age</th>       <td>   -1.0321</td> <td>    0.048</td> <td>  -21.524</td> <td> 0.000</td> <td>   -1.127    -0.938</td>
</tr>
<tr>
  <th>Education</th> <td>   -0.0099</td> <td>    0.252</td> <td>   -0.039</td> <td> 0.969</td> <td>   -0.507     0.487</td>
</tr>
<tr>
  <th>Gender</th>    <td>   -1.5948</td> <td>    1.597</td> <td>   -0.998</td> <td> 0.319</td> <td>   -4.746     1.557</td>
</tr>
<tr>
  <th>Student</th>   <td>  501.4411</td> <td>    2.587</td> <td>  193.814</td> <td> 0.000</td> <td>  496.337   506.546</td>
</tr>
<tr>
  <th>Married</th>   <td>   -2.5047</td> <td>    1.725</td> <td>   -1.452</td> <td> 0.148</td> <td>   -5.907     0.898</td>
</tr>
<tr>
  <th>Caucasian</th> <td>    0.1485</td> <td>    1.961</td> <td>    0.076</td> <td> 0.940</td> <td>   -3.721     4.018</td>
</tr>
<tr>
  <th>Asian</th>     <td>   -2.5407</td> <td>    2.280</td> <td>   -1.115</td> <td> 0.266</td> <td>   -7.038     1.957</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.399</td> <th>  Durbin-Watson:     </th> <td>   2.016</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.067</td> <th>  Jarque-Bera (JB):  </th> <td>   5.524</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.404</td> <th>  Prob(JB):          </th> <td>  0.0632</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.842</td> <th>  Cond. No.          </th> <td>5.92e+04</td>
</tr>
</table>




    modelb4 = sm.OLS(yb[xb.Rating <= 350],xb[xb.Rating <= 350]).fit()
    modelb4.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.851</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.842</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   99.66</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 17 Jun 2015</td> <th>  Prob (F-statistic):</th> <td>3.80e-73</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:07:02</td>     <th>  Log-Likelihood:    </th> <td> -1197.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   204</td>      <th>  AIC:               </th> <td>   2420.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   192</td>      <th>  BIC:               </th> <td>   2459.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>   
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
  <th>const</th>     <td> -218.3506</td> <td>   47.949</td> <td>   -4.554</td> <td> 0.000</td> <td> -312.924  -123.777</td>
</tr>
<tr>
  <th>Income</th>    <td>   -7.1002</td> <td>    0.446</td> <td>  -15.913</td> <td> 0.000</td> <td>   -7.980    -6.220</td>
</tr>
<tr>
  <th>Limit</th>     <td>    0.1523</td> <td>    0.043</td> <td>    3.563</td> <td> 0.000</td> <td>    0.068     0.237</td>
</tr>
<tr>
  <th>Rating</th>    <td>    0.4791</td> <td>    0.654</td> <td>    0.732</td> <td> 0.465</td> <td>   -0.811     1.769</td>
</tr>
<tr>
  <th>Cards</th>     <td>   13.9981</td> <td>    5.762</td> <td>    2.430</td> <td> 0.016</td> <td>    2.634    25.362</td>
</tr>
<tr>
  <th>Age</th>       <td>   -0.6226</td> <td>    0.368</td> <td>   -1.691</td> <td> 0.092</td> <td>   -1.349     0.104</td>
</tr>
<tr>
  <th>Education</th> <td>   -2.7730</td> <td>    2.072</td> <td>   -1.338</td> <td> 0.182</td> <td>   -6.859     1.313</td>
</tr>
<tr>
  <th>Gender</th>    <td>   25.9813</td> <td>   12.584</td> <td>    2.065</td> <td> 0.040</td> <td>    1.161    50.801</td>
</tr>
<tr>
  <th>Student</th>   <td>  339.5260</td> <td>   22.216</td> <td>   15.283</td> <td> 0.000</td> <td>  295.707   383.345</td>
</tr>
<tr>
  <th>Married</th>   <td>   -8.2622</td> <td>   12.799</td> <td>   -0.646</td> <td> 0.519</td> <td>  -33.507    16.983</td>
</tr>
<tr>
  <th>Caucasian</th> <td>   22.5267</td> <td>   15.427</td> <td>    1.460</td> <td> 0.146</td> <td>   -7.901    52.955</td>
</tr>
<tr>
  <th>Asian</th>     <td>   14.5170</td> <td>   17.628</td> <td>    0.824</td> <td> 0.411</td> <td>  -20.252    49.286</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.153</td> <th>  Durbin-Watson:     </th> <td>   1.919</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.207</td> <th>  Jarque-Bera (JB):  </th> <td>   2.562</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.155</td> <th>  Prob(JB):          </th> <td>   0.278</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.547</td> <th>  Cond. No.          </th> <td>2.49e+04</td>
</tr>
</table>




    stud_plot(modelb3)
    qq_plot(modelb3)
    lev_plot(modelb3,xb[xb.Rating>350].values)


![png](http://www.bryantravissmith.com/img/output_63_0.png)



![png](http://www.bryantravissmith.com/img/output_63_1.png)



![png](http://www.bryantravissmith.com/img/output_63_2.png)


The model for above 350 Credit Rating looks good.  Lets check below 350.


    stud_plot(modelb4)
    qq_plot(modelb4)
    lev_plot(modelb4,xb[xb.Rating<=350].values)


![png](http://www.bryantravissmith.com/img/output_65_0.png)



![png](http://www.bryantravissmith.com/img/output_65_1.png)



![png](http://www.bryantravissmith.com/img/output_65_2.png)


We have the same issue before.   We must made a cut to make a good prediction, then removed the values from the data that were causing issues without knowing the balance.   This can be useful in practice, but i find it a little unstatisfying.   For now we will stop with this exploration.  
