Title: Galvanize - Week 03 - Day 5
Date: 2015-06-19 10:20
Modified: 2015-06-19 10:30
Category: Galvanize
Tags: data-science, gradient decent, stocastic gradient decent, logistic regression
Slug: galvanize-data-science-03-05
Authors: Bryan Smith
Summary: Today we covered logistic regression and ROC curves.

#Galvanize Immersive Data Science

##Week 3 - Day 5

Today we had our checkin survey as a quiz, then we had a lecture on gradient decent.  We covered examples using linear and logistic regression.  The assignment was an all day paired sprint. 


##Gradient Descent

We will be implementing logistic regression using the gradient descent algorithm.  The goal in to include regulation and stocastic gradient decent.  We will start by testing it on data that will allow for simple solutions.


    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.optimize as op
    data = np.genfromtxt('data/testdata.csv', delimiter=',')
    X = data[:,0:2]
    y = data[:,2]
    
    xp = X[y==1,:]
    xn = X[y==0,:]
    plt.plot(xp[:,0],xp[:,1],'go',label="Positive")
    plt.plot(xn[:,0],xn[:,1],'ro',label="Negative")
    plt.xlabel("First Feature")
    plt.ylabel("Second Feature")
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D5/output_1_0.png)


This data set is very nice because the positive and negative examples are linearly separable.   I'll have to remember this dataset when I try to implement support vector machines.  I can guess the sigmoid function that will fit this data.

$$y = \frac{1}{1+e^{-2x_1}}$$


    x  = np.linspace(-6,8,100)
    y1 = 1/(1+np.exp(-2.*(x)))
    plt.plot(X[:,0],y,'ro')
    plt.plot(x,y1,'b-')
    plt.ylim([-0.1,1.1])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D5/output_3_0.png)


##Cost function

In order to be able to evaluate if our gradient descent algorithm is working correctly, we will need to be able to calculate the cost.  The cost function we will be using is the *log likelihood*. Our goal will be to *maximize* this value, so we will actually be implementing gradient *ascent*.

$$ \mathcal{l}(\Theta) = Log(\mathcal{L}(\Theta))= \Sigma_{i} ( \ y_i  \ log( \ h(x_i|\Theta) \ ) + (1-y_i) \ log( \ 1-h(x_i|\Theta) \ ) \ )$$

where the hypothesis fucntion is 

$$ h(x|\Theta) = \frac{1}{1 \ + \ e^{\Theta x}} $$

We will be using the gradient ascent potion of using 

$$\Theta_i = \Theta_i + \alpha \frac{\partial}{\partial\Theta_i} \mathcal{l}(\Theta)$$

Where alpha is the learning rate for the update.   We can show that 

$$\frac{\partial}{\partial\Theta_i} \mathcal{l}(\Theta) = \Sigma_{i} ( \ y_i \ - \ h(x_i|\Theta) \ ) \ x_j $$

We also implimented feature scaling, the options to fit the intercept, and Ridge (l2) penalizations.  This is done by subtracking a term from the cost function.

$$ \mathcal{l}(\Theta) = Log(\mathcal{L}(\Theta))= \Sigma_{i} ( \ y_i  \ log( \ h(x_i|\Theta) \ ) + (1-y_i) \ log( \ 1-h(x_i|\Theta) \ ) \ ) - \lambda \Theta^2$$

This changes the update function to


$$\frac{\partial}{\partial\theta_i} \mathcal{l}(\Theta) = \Sigma_{i} ( \ y_i \ - \ h(x_i|\Theta) \ ) \ x_j - 2 \ \lambda \theta_i$$

The larger the parameter lambda, the stronger the pull for the coefficients to be zero.

The last method we implmeneted was a stocastic gradient decent. Where we shuffled the data and take a set for each data point.  We then reshuffle the data and take another stuff.   This, in practice, is faster than hill climbing.   We did not implement this with regularization.   



    class regression_function:
        
        def __init__(self,x,y, fit_intercept=True, scale=False, lamb=0, tol=1e-5):
            self.x = x
            self.y = y
            self.lamb = lamb
            self.y.shape = (self.y.shape[0],1)
            self.theta = np.zeros((len(x[0,:]),1))
            self.scale = scale
            self.fit_intercept = fit_intercept
            self.tol = tol
            self.x_mean = self.x.mean(axis = 0)
            self.x_std = self.x.std(axis=0)
            if scale:    
                self.x = (self.x - self.x_mean)/self.x_std
            if fit_intercept:
                self.add_intercept()  
                
        def row_hypothesis(self,row):
            return 1/(1+np.exp(-self.x[row,:].dot(self.theta)))
    
    
        def hypothesis(self,X=None):
            if X == None:
                return 1/(1+np.exp(-self.x.dot(self.theta)))
            else:
                return 1/(1+np.exp(-X.dot(self.theta)))
            
        def predict(self,X=None,thresh=0.5):
            if X == None:
                return (self.hypothesis()>thresh).astype(int)
            else:
                if self.scale:
                    X = (X-self.x_mean)/self.x_std
                if self.fit_intercept:
                    X = self.add_intercept(X)
                return (self.hypothesis(X)>thresh).astype(int)
        
        def log_likelihood(self):
            llh = np.sum(self.y*np.log(self.hypothesis())+(1-self.y)*np.log(1-self.hypothesis()))-self.lamb*self.theta.T.dot(self.theta)
            return llh[0][0]
        
        def log_likelihood_gradient(self):    
            return self.x.T.dot(self.y-self.hypothesis())-2*self.lamb*self.theta
            
        
        def stoch_log_likelihood_gradient(self,row):
            xt = self.x[row].T.reshape(len(self.theta),1)
            return xt.dot(self.y[row] - self.row_hypothesis(row)).reshape(len(self.theta),1)
    
        def gradient_ascent(self,alpha):
            lik_diff = 1.
            previous_likelihood = self.log_likelihood()
            while lik_diff > self.tol:
                self.theta = self.theta + alpha*self.log_likelihood_gradient()
                temp = self.log_likelihood()
                lik_diff = temp - previous_likelihood
                previous_likelihood = temp
                
        def stoch_gradient_ascent(self,alpha=0.1):
            lik_diff = 1.
            previous_likelihood = self.log_likelihood()
            rows = range(len(self.x))
            np.random.shuffle(rows)
            self.x = self.x[rows,:]
            self.y = self.y[rows]
    
            while lik_diff > self.tol:
                for i in xrange(len(self.x)):
                    #print "STOCK LIKI:", self.stoch_log_likelihood_gradient(i)
                    #print "Shapes:", self.theta.shape,self.stoch_log_likelihood_gradient(i).shape
                    self.theta = self.theta + alpha*self.stoch_log_likelihood_gradient(i)
                temp = self.log_likelihood()
                #Sprint "TEMP:",temp
                lik_diff = temp - previous_likelihood
                previous_likelihood = temp    
                
                np.random.shuffle(rows)
                self.x = self.x[rows,:]
                self.y = self.y[rows]
    
        def add_intercept(self,X=None):
            if X==None:
                ones = np.ones((len(self.x),len(self.x[0,:])+1))
                ones[:,1:] = self.x
                self.x = ones
                self.theta = np.zeros((len(self.x[0,:]),1))
            else:
                ones = np.ones((len(X),len(X[0,:])+1))
                ones[:,1:] = X
                return ones
    
                
        def get_coeff(self):
            if self.scale:
                if self.fit_intercept:
                    return self.theta.T/np.hstack(([1],self.x_std))
        
                return self.theta.T/self.x_std 
            else:
                return self.theta.T
            
            


    r = regression_function(X,y)
    delta = .1
    t1 = np.arange(-3.0, 3.0, delta)
    t2 = np.arange(-3.0, 3.0, delta)
    T1, T2 = np.meshgrid(t1, t2)
    Z = T1.copy()
    for i,x1 in enumerate(t1):
        for j,y1 in enumerate(t2):
            r.theta[0,0] = x1
            r.theta[1,0] = y1
            Z[i,j] = r.log_likelihood()
    plt.figure()
    CS = plt.contour(T1, T2, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
    plt.title('Controur of Log Likelihood')




    <matplotlib.text.Text at 0x10a552c50>




![png](http://www.bryantravissmith.com/img/GW03D5/output_6_1.png)



    plt.pcolor(Z)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D5/output_7_0.png)


These plots are showing how seperable the data is.   We could fit this data with a number of logisitic function perfectly.  


    x  = np.linspace(-6,8,100)
    y1 = 1/(1+np.exp(-2.*(x)))
    y2 = 1/(1+np.exp(-10.*(x)))
    y3 = 1/(1+np.exp(-100.*(x)))
    plt.plot(X[:,0],y,'ro')
    plt.plot(x,y1,'b-')
    plt.plot(x,y2,'b-')
    plt.plot(x,y3,'b-')
    plt.ylim([-0.1,1.1])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW03D5/output_9_0.png)


So for any logistic function with a postive constant with a first feature will fit the data we have.  

##Compare to Sklearn


    r = regression_function(X,y)
    r.gradient_ascent(0.00001)
    print r.theta.transpose()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[ 0.01688462  1.27894634  0.00995178]]
    Accuracy:  1.0



    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    lin = LogisticRegression(fit_intercept=True)
    lin.fit(X,y)
    print lin.intercept_,lin.coef_,
    print accuracy_score(lin.predict(X),y)

    [ 0.04652751] [[ 1.2318561   0.02251709]] 1.0


    /Library/Python/2.7/site-packages/sklearn/utils/validation.py:449: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)


This give similar results to sklearn.   The sklearn package does seem to be faster, but I also believe it is using a c package called liblinear for its optimization, while we are doing it in pure python.

##Checking Scaling


    r = regression_function(X,y,fit_intercept=True, scale = True)
    r.gradient_ascent(0.0001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[ 0.17710665  1.17554584  0.05796619]]
    Accuracy:  1.0



    r = regression_function(X,y,fit_intercept=True, scale = False)
    r.gradient_ascent(0.00001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))
    print r.log_likelihood()

    [[ 0.01688462  1.27894634  0.00995178]]
    Accuracy:  1.0
    -0.227847971687



    r = regression_function(X,y,fit_intercept=True, scale = True)
    r.gradient_ascent(0.00001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[ 0.09595542  0.92911705  0.0366225 ]]
    Accuracy:  1.0


##Graduate Student Data


    data = np.genfromtxt('data/grad.csv', skiprows=1,delimiter=',')
    yg = data[:,0]
    xg = data[:,1:]
    rg= regression_function(xg,yg,fit_intercept=True,scale=True,tol=1e-6)
    rg.gradient_ascent(0.000001)
    print rg.get_coeff()
    np.sum(rg.predict()==rg.y)/float(len(rg.y))

    [[-0.84699307  0.00229446  0.76299305 -0.54843734]]





    0.70499999999999996




    log = LogisticRegression()
    log.fit(xg,yg)
    print np.hstack((log.intercept_,log.coef_[0]))
    from sklearn.metrics import accuracy_score
    print accuracy_score(log.predict(xg),yg)

    [-1.18847876  0.00191577  0.21564289 -0.59842009]
    0.715


    /Library/Python/2.7/site-packages/sklearn/utils/validation.py:449: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)


##Regularization


    r = regression_function(X,y,fit_intercept=True, scale = True,lamb=0)
    r.gradient_ascent(0.00001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[ 0.09595542  0.92911705  0.0366225 ]]
    Accuracy:  1.0



    r = regression_function(X,y,fit_intercept=True, scale = True,lamb=1)
    r.gradient_ascent(0.00001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[ 0.02568882  0.53308379  0.01324245]]
    Accuracy:  1.0



    r = regression_function(X,y,fit_intercept=True, scale = True,lamb=10)
    r.gradient_ascent(0.00001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[ 0.00178464  0.22617375  0.00074595]]
    Accuracy:  1.0



    r = regression_function(X,y,fit_intercept=True, scale = True,lamb=100)
    r.gradient_ascent(0.00001)
    print r.get_coeff()
    predictions = r.predict()
    print "Accuracy: ", np.sum(predictions==r.y)/float(len(r.y))

    [[  3.04588371e-06   4.27234980e-02  -8.65650480e-04]]
    Accuracy:  1.0


##Stochastic Gradient Descent


    r = regression_function(X,y,fit_intercept=True, scale = True,lamb=0, tol = 1e-5)
    print r.log_likelihood()
    %timeit r.stoch_gradient_ascent(11)
    print r.log_likelihood()
    print r.get_coeff()

    -69.314718056
    The slowest run took 68.69 times longer than the fastest. This could mean that an intermediate result is being cached 
    100 loops, best of 3: 2.29 ms per loop
    -0.000261667070057
    [[ 1.40603903  2.90379792  0.22544191]]



    r = regression_function(X,y,fit_intercept=True, scale = True,lamb=0, tol = 1e-5)
    print r.log_likelihood()
    %timeit r.gradient_ascent(1e-1)
    print r.log_likelihood()
    print r.get_coeff()

    -69.314718056
    The slowest run took 1034.52 times longer than the fastest. This could mean that an intermediate result is being cached 
    10000 loops, best of 3: 129 µs per loop
    -0.000393662821043
    [[ 1.31662671  2.79854521  0.25109632]]


##Newton's Method for a single variable
We were told to use newton's method for root finding on the following function:

$$f(x) = 6 \ x^2 + 3 \ x - 10$$

This function has two roots: -1.565 and 1.0650.


    
    import math
    def f(x):
        return 6*x**2+3*x-10
    
    def df(func,x):
        df = func(x+0.00001)-func(x-0.00001)
        return df/(0.00002)
    
    def newton_roots(func):
        tolerance = 1e-3
        xo = np.random.randint(-100,100)
        i = 0
        toll = 1
    
        while math.fabs(toll) > tolerance:
            xo = xo - func(xo)/df(func,xo)
            toll = func(xo)
            i += 1
            if i > 1e6:
                break
        return xo
    
    from collections import Counter
    
    cnt = Counter()
    
    for i in range(100):
        root = round(newton_roots(f),3)
        cnt[root] += 1
    print cnt

    Counter({1.065: 54, -1.565: 46})


We can see that we get the two roots about equal number of times.  
