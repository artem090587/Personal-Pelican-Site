Title: Galvanize - Week 04 - Day 5
Date: 2015-06-26 10:20
Modified: 2015-06-26 10:30
Category: Galvanize
Tags: data-science, galvanize, Profit Curves
Slug: galvanize-data-science-04-05
Authors: Bryan Smith
Summary: Today we covered Profit curves.


#Galvanize Immersive Data Science

##Week 4 - Day 5

Today we had an accessment on regression and classification methods we have covered the prevous two weeks.  It was a conceptional test, making sure we understood the underlying models and their applications.  

We then had a lecture on blogging.  Personally, I did not appreciate it.   I am obviously already blogging about what I am doing at Galvanize, but I also did not appreciate the frame given to the presentation: "I know you don't want to do this, but..."  It reminds me that we can all take issues based on attributes of an interaction that are not content based.  Good to be reminded of this going forward.

## Profit Curves

The afternoon sprint was on a profit curves.  You and read about them from chapter 8 of [Data Science For Business](http://www.amazon.com/Data-Science-Business-data-analytic-thinking/dp/1449361323).   

The goal of the assignment is to define a cost-benefit matrix for a business problem.  An example from the book involves calculating Lift, the increase in conversions.

$$ \left( \begin{array}{cc}  TP & FP \\ FN & TN \end{array} \right) => \left( \begin{array}{cc}  4 & -5 \\ 0 & 0 \end{array} \right) $$

In this case if we correct identify someone who will convert, we can spend mondy to convert them and make 5 dollars.   On the other hand, if we have a false positive and invest in someone who will not convert, we lose the 5 dollars of cost.   

If we do not predict someone to churn, we do not assume any cost.  But we also do not make any profit.   

In this case the best model will maximize True Positive and Minimize False Positives.   It needs high precisions, but not necessarily hight accuracy or recall.


The data set we are working with today is a cell phone dataset of users that churned.  We will start with the obligitory cleaning.   Since we have worked with this dataset before I will not be exploring it.  


    
    import matplotlib.pyplot as plt
    %matplotlib inline
    import numpy as np
    import pandas as pd
    
    churn = pd.read_csv('data/churn.csv').drop(['State','Area Code','Phone'],axis=1)
    churn.replace('yes',1, inplace=True)
    churn.replace('no',1,inplace=True)
    churn.replace('False.',0,inplace=True)
    churn.replace('True.',1,inplace=True)
    churn.head()




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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
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




    y = churn.pop("Churn?").values
    x = churn.values

Because we are dealing with a cell phone companies, the model is different.   I wil fillow suit with the Data Science For Buisness example and not worry about the fixed cost of the business.  Instead we will make a simple model that if we correcy identify someone who will churn then we will invest and make a profit.  If we incorrectly predict someone is going to churn and invest in keeping them, we lose the investment of cost.  

$$ \mbox{Profit Matrix} = \left( \begin{array}{cc}  80 & -20 \\ 0 & 0 \end{array} \right)$$


    profit_matrix = np.array([[80,-20],[0,0]])

The profit for a sample of uses will change if the total number of users changes.  For this reason we need make a rate to estimate the average profit per user.   We will have some model with a confusion matrix, and we will want to convert it a rate:


$$ \left( \begin{array}{cc}  TP & FP \\ FN & TN \end{array} \right) => \left( \begin{array}{cc}  \frac{TP}{TP+FP} & \frac{FN}{FN+TN} \\ \frac{FP}{TP+FP} & \frac{TN}{FN+TN} \end{array} \right) $$

This will allow us to get a feel for the acutal rate of misclassification and correct classification in the populations we are concerned with if we know the population proportions $P_+$ and $P_-$.

$$ \left( \begin{array}{cc}  \frac{TP}{TP+FP} \ P_+ & \frac{FN}{FN+TN} \ P_- \\ \frac{FP}{TP+FP} \ P_+ & \frac{TN}{FN+TN} \ P_- \end{array} \right)$$

In our dataset we have approximately 14% churn rate.  We can check that in the two extreams what will happen.   

If our model predicts that everyone will churn, our confusion matrix and rate look like:

$$ \left( \begin{array}{cc}  N_+ & N_- \\ 0 & 0 \end{array} \right) => \left( \begin{array}{cc}  1 & 1 \\ 0 & 0 \end{array} \right) $$

Since our $P_+ = .14$ and our $P_- = 0.86$, we have the error rate that looks like:

$$\left( \begin{array}{cc}  .14 & .86 \\ 0 & 0 \end{array} \right) $$

The element wise multipication with our cost matrix gives:

$$\left( \begin{array}{cc}  11.2 & -17.2 \\ 0 & 0 \end{array} \right) $$

We sum all the elements of this matrix together to get the expected profit:

$$E[\mbox{Profit}] = 11.2 - 17.2 + 0 + 0 = -6$$

This is obviously a bad strategy.  If we look at the other extreme and guess no one will churn we get the following results

$$ \left( \begin{array}{cc}  0 & 0 \\ N+ & N- \end{array} \right) => \left( \begin{array}{cc}  0 & 0 \\ 1 & 1 \end{array} \right) $$

$$\left( \begin{array}{cc}  0 & 0 \\  .14 & .86  \end{array} \right) $$

The element wise multipication with our cost matrix gives:

$$\left( \begin{array}{cc}  0 & 0 \\ 0 & 0 \end{array} \right) $$

$$E[\mbox{Profit}] = 0 + 0 + 0 + 0 = 0$$ 

So we have two extreme predictions.  One where we lose money, and one where we do not make a profit.   The idea is that if we can make smart prediction about who will churn and who will not churn, we can target our spending in the right places.   That will allow us to maximize our profit.   The smarter the predictions in terms of this problem and population, the better the results.  

##Smart Classifiers

We are going to make somem smart predictors.  By smart I mean genertic, untuned, machine learning algorithms.   We will try a Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting, and AdaBoost methods and compare their results.


    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split
    
    smart_classifiers = [LogisticRegression(),SVC(probability=True),GradientBoostingClassifier(), RandomForestClassifier(), AdaBoostClassifier()]


Because sklearn's confusion matrix calculation is in a different format that I was expecting, we built a helper function to calculate the rate and format it into the form we were expencting


    def confusion_rates(matrix):
        new = np.zeros(matrix.shape)
        new[0,0] = matrix[1,1]
        new[1,1] = matrix[0,0]
        new[1,0] = matrix[1,0]
        new[0,1] = matrix[0,1]
        return new.astype(float)/np.sum(new,axis=0)
    
    confusion_rates(np.array([[94,6],[96,4]]))




    array([[ 0.04,  0.06],
           [ 0.96,  0.94]])



##Profit Curves Theory

The idea behind the profit curves we are producing is that each classifier is fitted to training data.   Then test data is given to the classifier, the classifier predicts if it is a positive example or negative example.   The classifieres we are testing also estimate how strongly the algorithm believes each instance is a positive example (or negative example).    

We can now take the strength of these believes and asked the question:  If we only take the test example with the strongest belief as a positive example, and guess that all other test examples as negative example, how profitable is this. We should get something close to zero in our churn example.

We can then ask the question for the two strongest predictions, then the three strongest predictions, and so on.  We do this until we just guess everything is a positive example.   That would lead us back to the -6 dollar profits in our churn model.  

###Example

$$truth = [1, \ 1, \ 0, \ 0, \ 1]$$
$$ \mbox{model 1} = [0.9, \ 0.4, \ 0.2, \ 0.8, \ 0.7] $$
$$\mbox{model 2} = [0.8, \ 0.7, \ 0.2, \ 0.3, \ 0.6] $$


Model 1 and Model 2 have the same strongest predictor, so we would precict the following for our first question:

$$ \mbox{model 1} = [1, \ 0, \ 0, \ 0, \ 0]$$
$$ \mbox{model 2} = [1, \ 0, \ 0, \ 0, \ 0]$$

Leading to the confusion matrix:

$$ \left( \begin{array}{cc}  1 & 0 \\ 2 & 1 \end{array} \right) => \left( \begin{array}{cc}  0.33 & 0 \\ 0.67 & 1 \end{array} \right) $$

We have the proportion of positive examples $P_+ = 0.6$ and $P_- = 0.4$.  That leads to the classification rates of:

$$\left( \begin{array}{cc}  0.2 & 0 \\ 0.4 & 0.4 \end{array} \right) $$

Using the cost matrix from the Data Science for Business School of 4 dollars profit for true positives and -5 dollars profit for false positives we get a profit matrix of:

$$\left( \begin{array}{cc}  0.8 & 0 \\ 0 & 0 \end{array} \right) $$

For if we only take the strongest predictor we expect an average profit of 0.8 per costomer for both models.

$$E(\mbox{Profit}) = 0.8 + 0 + 0 + 0 = 0.8$$

We can now ask the second question of what is the expected profit if we take two two strongest predictors.  In this case we have the following predecitions:

$$ \mbox{model 1} = [1, \ 0, \ 0, \ 1, \ 0]$$
$$ \mbox{model 2} = [1, \ 1, \ 0, \ 0, \ 0]$$

Now the two models make different predictions!

The confusion matrix for model 1 (left) and model 2 (right) right are next. 

$$ \left( \begin{array}{cc}  1 & 1 \\ 2 & 1 \end{array} \right) \ ,  \ \left( \begin{array}{cc}  2 & 0 \\ 1 & 2 \end{array} \right) $$

The rate matrixes become:

$$ \left( \begin{array}{cc}  0.33 & .5 \\ .67 & .5 \end{array} \right) \ , \ \left( \begin{array}{cc}  0.67 & 0 \\ 0.33 & 1 \end{array} \right) $$

The population proportions have not changed. The proportion of positive examples $P_+ = 0.6$ and $P_- = 0.4$. 

$$\left( \begin{array}{cc}  0.2 & 0.2 \\ 0.4 & 0.2 \end{array} \right) \ , \ \left( \begin{array}{cc}  0.4 & 0 \\ 0.2 & 0.4 \end{array} \right) $$

That then makes the final profit matrix look like:

$$\left( \begin{array}{cc}  0.8 & -1 \\ 0 & 0 \end{array} \right) \ , \ \left( \begin{array}{cc}  1.6 & 0 \\ 0 & 0 \end{array} \right)  $$

Now there is a clear difference in the models if we take the two strongest predictors.

$$E(\mbox{Profit Model 1}) = 0.8 + -1 + 0 + 0 = -0.2$$

$$E(\mbox{Profit Model 2}) = 1.6 + 0 + 0 + 0 = 1.6$$

Model 2 is much better model if we limit ourselves to the two strongest predictions.  To find the most profitable model, we should conditinue to do these calcuations.  Becausse they are so repetative, its time for some automation

##Plotting Profit Curves


    def profit_curve(classifiers, cb, x, y):
        #split data into a training and test set
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35)
        
        #get the true proportions in the test set
        p_pos = np.sum(y_test==1).astype(float)/len(y_test)
        p_neg = 1 - p_pos
        
        #scale the training data - important for some classifiers and to get consistent results
        s = StandardScaler()
        xtrn = s.fit_transform(x_train)
        xtst = s.transform(x_test)
    
        for c in classifiers:
            prob = c.fit(xtrn,y_train).predict_proba(xtst)[:,1]
            
            #Get the indexes of the data most likely to be positive
            indicies = np.argsort(prob)[::-1]
            costs = []
            x_axis = []
            #For each data point
            for i in indicies:
                
                #predict the all probabilities above the ith strongest predicters is positive, else negative
                y_pred = (prob > prob[i]).astype(int)
                 
                #calculate the confusion matrix for the predictions
                matrix = confusion_rates(confusion_matrix(y_test,y_pred))
                #matrix = np.nan_to_num(matrix)
                
                #calculate the cost matrix through element wise product and sum
                cost = np.sum((matrix*cb).dot(np.array([[p_pos],[p_neg]])))
                
                #append the cost and proportion of test predictions we set positive
                x_axis.append(np.sum(y_pred).astype(float)/len(y_pred))
                costs.append(cost)
    
            p = prob[indicies[np.argsort(costs)[::-1][0]]]
            
            plt.plot(x_axis, costs, label=c.__class__.__name__)           



    plt.figure(figsize=(14,14))
    profit_curve(smart_classifiers,profit_matrix,x, y)
    plt.legend()
    plt.ylabel('Expected Profit Rate')
    plt.xlabel('% Test Predictions Set Positive')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D5/output_14_0.png)


##Analysis

The plots we have generated match our intuition we developed.  If we do not guess anyone churns (left), then we expect the profit to be zero.   If we guess everyone churns (right), we expect the profit to be -6.  The acutal value is different be the actual proportion in the test set is different from 14%.  Inbetween we are using smart classifier to predict who will churn and who will not churn.  Even the worst model leads to some profitability, which is promising for our cell phone company.   

The best model, in this trial, is the GradientBoostingClassifier, followed closely by the Support Vector Machine and Randome Forest.  The peak profitability for all the models seem to be taking the 18% of strongest predictors in the models.  


    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35)
    
    #Prediction Probabilities on the test set
    probs = GradientBoostingClassifier().fit(x_train,y_train).predict_proba(x_test)[:,1]
    
    #class probabilities ~ [.14,.86]
    class_prob = np.array([sum(y_test).astype(float)/len(y_test), sum(y_test==0).astype(float)/len(y_test) ])
    
    #get the cutoff for the tope 18%, 100-18 - 82
    percentile = np.percentile(probs,82)
    print "If probabibility is larger than this predict Churn:", percentile
    y_pred = (probs >= percentile).astype(int)
    rates = confusion_rates(confusion_matrix(y_test,y_pred))
    
    print "Predicted Profit Rate:", np.sum(rates*class_prob*profit_matrix)
    
    print "% Costomers Affected/Targed, %of Correct Costomers:", np.sum((rates*class_prob)[0]),(rates*class_prob)[0,0]

    If probabibility is larger than this predict Churn: 0.184275129582
    Predicted Profit Rate: 6.51242502142
    % Costomers Affected/Targed, %of Correct Costomers: 0.179948586118 0.101113967438


For our gradient model we will predict a costomer to be likely churn if their prediciton probability is above 0.18, and we have expected profit rate of $6.5/prediction.   We estimate that we will target approximately 18% of our costomers using this model, with 10% of them being the group we want to target.

If we have an unlimited budget, or a budget larger than the cost of targeting every costomer, then we would choose this model.  

If we have a limited budget that want to be profitable (near term) but are willing to not persue maximal profit to reach the largest number of customers and reduce churn, then we will want a differnet strategy.  Because the cost of targeting a customer is fixed, we will want to optimize precision by moving to the left of the graph.  In our case of the cell phone churn, the the top models move in lock-step.   

We would just increase the threshold for our GradientBoostedClassifier.  If we increase the threshold to be the top 10% of predictions instead of the top 18%, we get reduce profitability but increase targeting rate.


    #Prediction Probabilities on the test set
    probs = GradientBoostingClassifier().fit(x_train,y_train).predict_proba(x_test)[:,1]
    
    #class probabilities ~ [.14,.86]
    class_prob = np.array([sum(y_test).astype(float)/len(y_test), sum(y_test==0).astype(float)/len(y_test) ])
    
    #get the cutoff for the tope 18%, 100-18 - 82
    percentile = np.percentile(probs,92)
    print "If probabibility is larger than this predict Churn:", percentile
    y_pred = (probs >= percentile).astype(int)
    rates = confusion_rates(confusion_matrix(y_test,y_pred))
    
    print "Predicted Profit Rate:", np.sum(rates*class_prob*profit_matrix)
    
    print "% Costomers Affected/Targed, %of Correct Costomers:", np.sum((rates*class_prob)[0]),(rates*class_prob)[0,0]

    If probabibility is larger than this predict Churn: 0.5623562992
    Predicted Profit Rate: 5.8440445587
    % Costomers Affected/Targed, %of Correct Costomers: 0.0805484147386 0.0745501285347


Here, our expected profit drops to 5.8 from 6.5, and our threshold is increased to 0.56.   What we like to see is now we are only targeting 8% of our customers, but 7.5% of them are the ones we want to target.  That takes our precision to approximately 93% from 56%.  We are not getting the most profit, but we are spending our money in the most targeted way in this campaign.  If we know what percentage of our customers we can target with our budget, we can move down the curve and clacluate our expected results.  


    10.1/17.99, 0.07455/0.08055




    (0.5614230127848805, 0.9255121042830541)




    
