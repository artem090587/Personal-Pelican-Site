Title: Something Else
Date: 2010-12-03 10:22
Modified: 2010-12-05 19:32
Category: Python
Tags: pelican, publishing
Slug: something-else
Authors: Bryan Smith
Summary: Short version for index and feeds
Email: bryantravissmith@gmail.com
About_author: About Me!!!!

<figure>
  <IMG SRC="https://raw.githubusercontent.com/mbakker7/exploratory_computing_with_python/master/tudelft_logo.png" WIDTH=250 ALIGN="right">
</figure>

# Exploratory Computing with Python
*Developed by Mark Bakker*
## Advanced Topic Notebook 1: Finding the zero of a function

Finding the zero of a function is a very common task in exploratory computing. In mathematics it is also called *root finding*. The `scipy` package contains a number of methods to find the (approximate) value of the zero of a function of one or more variables. In this Notebook, we will program two methods ourselves, the Bisection method and Newton's method. At the end of the Notebook, we use the corresponding functions of `scipy` to obtain the same results.


    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

###Bisection method
The Bisection method is a simple method to find the zero of a function. The user needs to specify the function $f(x)$ and two values of $x$ between which $f(x)$ is zero - let's call those two points $x_1$ and $x_2$. As $f(x)$ equals zero somewhere between $x_1$ and $x_2$, that means that $f(x)$ is positive at $x_1$ or $x_2$ and negative at the other one. In other words, the product of the two function values is negative: $f(x_1)f(x_2)<0$. If this condition is fulfilled, all we know is that $f(x)$ is zero somewhere in the interval between $x_1$ and $x_2$ (provided $f(x)$ is continuous, of course). The basic idea of the bisection method is to iterate towards the zero of the function by cutting the interval in half every iteration. This is done by computing the middle between $x_1$ and $x_2$, let's call that point $x_m$, and compute $f(x_m)$. Next, replace either $x_1$ or $x_2$ by $x_m$ making sure that $f(x)$ remains negative at one of the two values and positive at the other. Repeat the process until the interval is small enough. In summary, the algorithm works as follows:

1. Compute $f(x)$ at $x_1$ and $x_2$ and make sure that $f(x_1)f(x_2)<0$.
2. Compute $x_m = \frac{1}{2}(x_1+x_2)$.
3. Compute $f(x_m)$.
4. If $f(x_m)f(x_2)<0$, replace $x_1$ by $x_m$, otherwise, replace $x_2$ by $x_m$.
5. Test whether $|x_1-x_2|<\varepsilon$, where $\varepsilon$ is a user-specified tolerance. If this is not yet the case, return to step 2.

Recall that a function may simply be passed as the argument to another function in Python. The example below contains a function called `square_me` that returns the square of any function of one variable, evaluated at the provided value of $x$. As an example, `square_me` is used with the `cos` function


    def square_me( func, x ):
        return func(x)**2
    print 'result of square_me function: ',square_me( np.cos, 4 )
    print 'directly taking the square  : ',np.cos(4)**2

    result of square_me function:  0.427249983096
    directly taking the square  :  0.427249983096


###Exercise 1. <a name="back1"></a>Bisection method

*Step 1.*
Write a Python function for $f(x)=\frac{1}{2}-\text{e}^{-x}$. Create a plot of $f(x)$ for $x$ varying from 0 to 4. Notice that $f(x)$ has a zero somewhere on the plotted interval (for this example it isn't really that hard to determine the zero exactly, of course, and we will do that later on to test whether our code works correctly).


    

*Step 2.* Implement the bisection method in a function called `bisection`. Your `bisection` method should take the following arguments:

1. The function for which you want to find the zero.
2. $x_1$ and $x_2$
3. The tolerance `tol` used as a stopping criterion. Make `tol` a keyword argument with a default value of 0.001.
4. The maximum number of iterations `nmax`. Make `nmax` a keyword argument with a default value of, for example, 10.

Your function should return the value of $x$ where $f(x)$ equals (approximately) zero. Your function should print a warning to the screen when the maximum number of iterations is reached before the tolerance is met.

In writing your code, implement steps 2-5 of the algorithm explained above in a regular loop that you perform `nmax` times and break out of the loop (using the `break` command) when the tolerance is met. Doing it this way will prevent you from getting stuck in an infinite loop, which may happen if you use a `while` loop. 
In writing your code it is advisable to print the values of $x_1$, $x_2$, $f(x_1)$, and $f(x_2)$ to the screen every iteration, so you can see how your `bisection` function performs (or whether you have any bugs left). 

Use your `bisection` method to find the zero of the function $f(x)$ you programmed in Step 1 and make sure it is within `tol=0.001` of the exact value (which you can determine by hand). 


    

*Step 3*
Demonstrate that your `bisection` method works correctly by finding the zero of cos($x$) between $x_1=0$ and $x_2=3$ running the following command:

`bisection(np.cos, 0, 3, tol=1e-6, nmax=30)`


    

<a href="#ex1answer">Answers to Exercise 1</a>

### Newton's method

The Bisection method is a brute-force method that is guaranteed to work when the user specifies an interval from $x_1$ to $x_2$ that contains a zero (and the function is continuous). The Bisection method is not very efficient (it requires a lot of function evaluations) and it is inconvienient that the user has to specify an interval that contains the zero. A smarter alternative is Newton's method (also called the Newton-Raphson method), but it is, unfortunately, not guaranteed that it always works, as is explained below. 

Let's try to find the zero of the function represented by the blue line in the graph below. Newton's method starts at a user-defined starting location, called $x_0$ here and shown with the blue dot. A straight line is fitted through the point $(x,y)=(x_0,f(x_0))$ in such a way that the line is tangent to $f(x)$ at $x_0$ (the red line). The intersection of the red line with the horizontal axis is the next estimate $x_1$ of the zero of the function (the red dot). This process is continued until a value of $f(x)$ is found that is sufficiently small. Hence, a straight line is fitted through the point $(x,y)=(x_1,f(x_1))$, again tangent to the function, and the intersection of this line with the horizontal axis is the next best estimate of the zero of the function, etc., etc.


<img src="http://i.imgur.com/tK1EOtD.png" alt="Newton's method on wikipedia">

The equation for a straight line with slope $a$ and through the point $x_n,f(x_n)$ is

$y = a(x-x_n) + f(x_n)$

As we want the line to be tangent to the function $f(x)$ at the point $x=x_n$, the slope $a$ is equal to the derivative of $f(x)$ at $x_n$: $a=f'(x_n)$. To find the intersection of the line with the horizontal axis, we have to find the value of $x$ that results in $y=0$. This is our next estimate $x_{n+1}$ of the zero of the function. Hence we need to solve

$0 = f'(x_n) (x_{n+1}-x_n) + f(x_n)$

which gives

$\boxed{x_{n+1} = x_n - f(x_n)/f'(x_n)}$

The search is completed when $|f(x)|$ is below a user-specified tolerance.
A nice animated illustration of Newton's method can be found on wikipedia (don't worry, we'll learn how to make cool animations like this as well).

<img src="http://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif" alt="Newton's method on wikipedia" width="400px">

Newton's method is guaranteed to find the zero of a function if the function is *well behaved* and you start your search *close enough* to the zero. If those two conditions are met, Newton's method is very fast. If they are not met, the method does not converge to the zero. Another disadvantage of Newton's method is that you need to define the derivative of the function. Strangely enough, the function value doesn't have to go down every iteration (as is illustated in the figure above going from $x_2$ to $x_3$).

###Exercise 2. <a name="back2"></a>Newton's method
Implement Newton's method in a Python function called `newtonsmethod` and test your function by finding the zero of $f(x)=\frac{1}{2}-\text{e}^{-x}$, as we used in Exercise 1. Use $x_0=1$ as the starting point of your search. The `newtonsmethod` function should take the following arguments:

1. The function for which you want to find the zero.
2. The derivative of the function for which you want to find the zero.
3. The starting point of the search $x_0$.
4. The tolerance `tol` used as a stopping criterion. Make `tol` a keyword argument with a default value of $10^{-6}$.
5. The maximum number of iterations `nmax`. Make `nmax` a keyword argument with a default value of 10.

Your function should return the value of $x$ where $f(x)$ equals (approximately) zero. Your function should print a warning to the screen when the maximum number of iterations is reached before the tolerance is met. 

It is suggested to print the value of $x_{n+1}$ and the corresponding function value to the screen every iteration so you know whether your function is making good progress. If you implement everything correctly, you should find a zero that gives a function value less than $10^{-6}$ within 3 iterations if you start at $x=1$. How many iterations do you need when you start at $x=4$?


    

Demonstrate that your `newton` function works by finding the zero of $\sin(x)$. As you know, the $\sin(x)$ function has many zeros: $-2\pi$, $-\pi$, $0$, $pi$, $2\pi$, etc. Which zero do you find when starting at $x=1$ and which zero do you find when starting at $x=1.5$?


    

<a href="#ex2answer">Answers to Exercise 2</a>

###Root finding methods in `scipy`
The package `scipy.optimize` includes a number of routines for the minimization of a function and to find the zeros of a function. Two of the rootfinding methods are called, no suprise, `bisect` and `newton`. 

###Exercise <a name="back3"></a>3
Use the `newton` method of `scipy.optimize` package to find the $x$ value for which $\ln(x^2)=2$ (i.e., find the zero of the function $\ln(x^2)-2$), and demonstrate that your value of $x$ indeed gives $\ln(x^2)=2$.


    

###Optimization

###Answers to the exercises

<a name="ex1answer">Answers to Exercise 1</a>


    def f(x):
        return 0.5 - np.exp(-x)
    
    x = np.linspace(0,4,100)
    y = f(x)
    plt.plot(x,y)
    plt.axhline(0,color='r',ls='--')




    <matplotlib.lines.Line2D at 0x1063958d0>




![png](output_27_1.png)



    def bisection(func, x1, x2, tol=1e-3, nmax=10):
        f1 = func(x1)
        f2 = func(x2)
        assert f1*f2< 0, 'Error: zero not in interval x1-x2'
        for i in range(nmax):
            xm = 0.5*(x1+x2)
            fm = func(xm)
            if fm*f2 < 0:
                x1 = xm
                f1 = fm
            else:
                x2 = xm
                f2 = fm
            print x1,x2,f1,f2
            if abs(x1-x2) < tol:
                return x1
        print 'Maximum number of iterations reached'
        return x1
        
    xzero = bisection(f,0,4,nmax=20)  
    print 'zero of function and function value: ',xzero,f(xzero)  

    0 2.0 -0.5 0.364664716763
    0 1.0 -0.5 0.132120558829
    0.5 1.0 -0.106530659713 0.132120558829
    0.5 0.75 -0.106530659713 0.027633447259
    0.625 0.75 -0.035261428519 0.027633447259
    0.6875 0.75 -0.00283157797094 0.027633447259
    0.6875 0.71875 -0.00283157797094 0.0126389232864
    0.6875 0.703125 -0.00283157797094 0.0049641030738
    0.6875 0.6953125 -0.00283157797094 0.00108148841353
    0.69140625 0.6953125 -0.000871223429674 0.00108148841353
    0.69140625 0.693359375 -0.000871223429674 0.000106085964203
    0.6923828125 0.693359375 -0.000382330131828 0.000106085964203
    zero of function and function value:  0.6923828125 -0.000382330131828



    xzero = bisection(np.cos, 0, 3, tol=1e-6, nmax=30)
    print 'cos(x) is zero between 0 and pi at:',xzero
    print 'relative error:',(xzero-np.pi/2)/(np.pi/2)

    1.5 3 0.0707372016677 -0.9899924966
    1.5 2.25 0.0707372016677 -0.628173622723
    1.5 1.875 0.0707372016677 -0.29953350619
    1.5 1.6875 0.0707372016677 -0.116438941125
    1.5 1.59375 0.0707372016677 -0.0229516576536
    1.546875 1.59375 0.0239190454439 -0.0229516576536
    1.5703125 1.59375 0.00048382677602 -0.0229516576536
    1.5703125 1.58203125 0.00048382677602 -0.0112346868547
    1.5703125 1.576171875 0.00048382677602 -0.00537552231604
    1.5703125 1.5732421875 0.00048382677602 -0.00244585826649
    1.5703125 1.57177734375 0.00048382677602 -0.000981016797749
    1.5703125 1.57104492188 0.00048382677602 -0.000248595077543
    1.57067871094 1.57104492188 0.000117615857125 -0.000248595077543
    1.57067871094 1.57086181641 0.000117615857125 -6.54896113066e-05
    1.57077026367 1.57086181641 2.60631230187e-05 -6.54896113066e-05
    1.57077026367 1.57081604004 2.60631230187e-05 -1.97132441646e-05
    1.57079315186 1.57081604004 3.17493942786e-06 -1.97132441646e-05
    1.57079315186 1.57080459595 3.17493942786e-06 -8.26915236891e-06
    1.57079315186 1.5707988739 3.17493942786e-06 -2.54710647057e-06
    1.57079601288 1.5707988739 3.1391647865e-07 -2.54710647057e-06
    1.57079601288 1.57079744339 3.1391647865e-07 -1.11659499596e-06
    1.57079601288 1.57079672813 3.1391647865e-07 -4.01339258654e-07
    cos(x) is zero between 0 and pi at: 1.57079601288
    relative error: -1.99845437142e-07


<a href="#back1">Back to Exercise 1</a>

<a name="ex2answer">Answers to Exercise 2</a>


    def f(x):
        return 0.5 - np.exp(-x)
        
    def fp(x):
        return np.exp(-x)
    
    def newtonsmethod(func, funcp, xs, tol=1e-6, nmax=10):
        f = func(xs)
        for i in range(nmax):
            fp = funcp(xs)
            xs = xs - f/fp
            f = func(xs)
            print xs,func(xs)
            if abs(f) < tol: 
                print 'tolerance reached in',i+1,'iterations'
                break
        if abs(f) > tol:
            print 'Max number of iterations reached before convergence'
        return xs
        
    print 'starting at x=1'
    xzero = newtonsmethod(f,fp,1)
    print 'xzero,f(xzero) ',xzero,f(xzero)
    
    print 'starting at x=4'
    xzero = newtonsmethod(f,fp,4,nmax=40)
    print 'xzero,f(xzero) ',xzero,f(xzero)

    starting at x=1
    0.64085908577 -0.0268396291473
    0.691803676235 -0.000672203615638
    0.693146278462 -4.5104915336e-07
    tolerance reached in 3 iterations
    xzero,f(xzero)  0.693146278462 -4.5104915336e-07
    starting at x=4
    -22.2990750166 -4834652137.25
    -21.2990750167 -1778569126.38
    -20.299075017 -654299016.164
    -19.2990750177 -240703156.293
    -18.2990750198 -88549742.4933
    -17.2990750254 -32575629.6522
    -16.2990750408 -11983904.3001
    -15.2990750825 -4408631.88486
    -14.2990751959 -1621844.90201
    -13.2990755042 -596643.264099
    -12.2990763422 -219492.658455
    -11.2990786202 -80746.7044133
    -10.2990848124 -29704.920376
    -9.29910164433 -10927.697389
    -8.29914739753 -4019.94309239
    -7.29927176193 -1478.72230925
    -6.29960977738 -543.859447306
    -5.30052828812 -199.942673493
    -4.30302276693 -73.4229080399
    -3.3097865701 -26.879281291
    -2.32804855421 -9.75790424455
    -1.37679145414 -3.4621684118
    -0.502984979508 -1.15365002236
    0.194653581712 -0.323119752408
    0.587208554986 -0.0558768205856
    0.687728703544 -0.00271659175616
    0.693132527092 -7.32678775317e-06
    0.693147180453 -5.36808375529e-11
    tolerance reached in 28 iterations
    xzero,f(xzero)  0.693147180453 -5.36808375529e-11



    xzero = newtonsmethod(np.sin, np.cos, 1)
    print 'starting point is x=1'
    print 'xzero,sin(xzero) ', xzero, np.sin(xzero)
    
    xzero = newtonsmethod(np.sin, np.cos, 1.5)
    print 'starting point is x=1.5'
    print 'xzero,sin(xzero) ', xzero, np.sin(xzero)
    print 'xzero / pi ', xzero/np.pi

    -0.557407724655 -0.52898809709
    0.0659364519248 0.0658886845842
    -9.57219193251e-05 -9.57219191789e-05
    2.92356620141e-13 2.92356620141e-13
    tolerance reached in 4 iterations
    starting point is x=1
    xzero,sin(xzero)  2.92356620141e-13 2.92356620141e-13
    -12.6014199472 -0.035042157161
    -12.5663562551 1.43592405006e-05
    -12.5663706144 -1.28649811974e-15
    tolerance reached in 3 iterations
    starting point is x=1.5
    xzero,sin(xzero)  -12.5663706144 -1.28649811974e-15
    xzero / pi  -4.0


<a href="#back2">Back to Exercise 2</a>

<a name="ex3answer">Answers to Exercise 3</a>


    import scipy.optimize as so
    
    def g(x):
        return np.log(x**2)-2
    
    x = so.newton(g,1)
    print 'value of x:', x
    print 'ln(x^2):', np.log(x**2)

    value of x: 2.71828182846
    ln(x^2): 2.0


<a href="#back3">Back to Exercise 3</a>
