Title: Galvanize - Week 04 - Day 1
Date: 2015-06-22 10:20
Modified: 2015-06-2 10:30
Category: Galvanize
Tags: data-science, galvanize, kNN, K Nearest Neighbors, Decision Trees
Slug: galvanize-data-science-04-01
Authors: Bryan Smith
Summary: Today we covered K Nearest Neighbors and Decision Trees
#Galvanize Immersive Data Science

##Week 4 - Day 1

Our quiz involved creating an SQL table that will determine churn for an fake web adervertising data.  Then with this new table, and a table of predictions, we had to come up with SQL queries that would calculate accuracy, precision, recall, and specificity.  
   
###Tables
    advertisers
        id
        name
        city
        state
        business_type

    campaigns
        advertiser_id
        campaign_id
        start_date
        duration
        daily_budget
        
    predicted_churn
        advertiser_id
        churn

###Table Query
	CREATE TABLE churn 
	AS (SELECT a.id, 
			a.name, 
			a.city, 
			a.state, 
			a.business_type, 
			(DATEDIFF(day,
				GETDATE(),
				c.start_date)
				+c.duration > 14) AS churn
		FROM advertisers a 
		JOIN (SELECT c.* 
				FROM campaigns c 
				JOIN (SELECT advertiser_id, 
						MAX(start_date) as last_campaign_date 
			 			GROUPBY advertiser_id) cc 
				ON c.advertiser_id = cc.advertiser
				AND c.start_date = cc.last_campaign_date) c
		ON a.id = c.advertiser_id
		);

###Metric Query

    SELECT CAST((TP+TN) AS FLOAT)/(TP+TN+FP+FN) as accuracy,
				CAST(TP AS FLOAT)/(TP+FN) as recall,
				CAST(TP AS FLOAT)/(TP+FP) as precision,
				CAST(TN AS FLOAT)/(TN+FP) as specificity
    FROM (SELECT COUNT( CASE WHERE c.churn=1 AND pc.churn=1
                              THEN 1 ELSE 0) as TP,
                 COUNT( CASE WHERE c.churn=1 AND pc.churn=0
                              THEN 1 ELSE 0) as FN,
                 COUNT( CASE WHERE c.churn=0 AND pc.churn=1
                              THEN 1 ELSE 0) as FP,
                 COUNT( CASE WHERE c.churn=0 AND pc.churn=0
                              THEN 1 ELSE 0) as TN,
          FROM churn c JOIN predicted_churn)


##k Nearest Neighbors

Our morning individual sprit was to implement a kNN class that can take differnt similarity functions as a measure of nearest.  For good for bad, mine was by var the tursted solution, but that is because I perfer to think interms of matrix operations.  Because of my use of numpy, mine was also the fastest!


We will start with loading some data.



    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    
    from sklearn.datasets import make_classification,load_iris
    X, y = make_classification(n_features=4, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, class_sep=5, random_state=5)


    def euclidean_distance(row,arr):
        '''
            INPUT:
                - row: 1d numpy array k x 1
                - arr: 2d numpy array m x n
            OUTPUT:
                - 1d numpy array m x 1
            
            Calculates the euclidean distance of row from each row in arr.
        '''
        
        return np.sum(np.power(row-arr,2),axis=1)
    
    def euclidean_distance(row,arr):
        '''
            INPUT:
                - row: 1d numpy array k x 1
                - arr: 2d numpy array m x n
            OUTPUT:
                - 1d numpy array m x 1
            
            Calculates the euclidean distance of row from each row in arr.
        '''
        
        return np.sum(np.power(row-arr,2),axis=1)
    
    def cosine_distance(row,arr):
        '''
            INPUT:
                - row: 1d numpy array k x 1
                - arr: 2d numpy array m x n
            OUTPUT:
                - 1d numpy array m x 1
            
            Calculates the cosign similarity of row from each row in arr.
            cosign similarity = 1 - a.dot(b)/(|a||b|)
        '''
        return 1-arr.dot(row)/np.linalg.norm(row)/np.linalg.norm(arr,axis=1)


    class KNN:
        
        def __init__(self,k,similarity):
            '''
            INPUT:
                - k: int > 0
                - similarity: function(1d numpy array,2d numpy array) returns 1d numpy array
            OUTPUT:
                - None
            
            Instantiates the KNN class
            '''
            
            self.k = k
            self.similarity = similarity
            
        def fit(self,X,y):
            '''
            INPUT:
                - X: 2d numpy array
                - y: 1d numpy array of labels
            OUTPUT:
                - None
            
            Stores training data
            '''
            self.X = X
            self.y = y
            
        def predict(self,X):
            '''
            INPUT:
                - X: 2d numpy array
            OUTPUT:
                - 1d numpy array
            
            Calculate the distances of each row in X from each row in the training data.  
            Returns the max vote from k nearest points.
            '''
            distances = np.apply_along_axis(lambda x:self.similarity(x,self.X),1,X)
            return np.apply_along_axis(lambda x:np.argmax(np.bincount(self.y[np.argsort(x)[:self.k]])),1,distances)
            
            
        def score(self):
            '''
            INPUT:
            
            OUTPUT:
                - numpy float
            
            Accurcy of kNN on training data
            '''
            return np.sum(self.predict(self.X)==self.y)/float(len(self.y))


    knn = KNN(10,euclidean_distance)
    knn.fit(X,y)
    knn.predict(X)
    knn.score()




    1.0




    knn = KNN(10,cosine_distance)
    knn.fit(X,y)
    knn.predict(X)
    knn.score()




    1.0



##kNN on Iris Data

I will apply my KNN class on the Iris dataset from sklearn, and compare my results to the sklearn results.  I will do this for both my metrics of euclidean and cosign.  


    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    n_neighbors = 5
    iris = load_iris()
    XI = iris.data[:, :2] 
    yI = iris.target
    
    for func in [euclidean_distance, cosine_distance]:
        knn = KNN(n_neighbors,func)
        knn.fit(XI,yI)
    
        h = 0.2
        x_min, x_max = XI[:, 0].min() - 1, XI[:, 0].max() + 1
        y_min, y_max = XI[:, 1].min() - 1, XI[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
        plt.scatter(XI[:, 0], XI[:, 1], c=yI, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Bryan " + func.__name__ + " KNN 3-Class classification (k = %i)" % (n_neighbors))
    
    
        from sklearn.neighbors import KNeighborsClassifier as sKNN
        sknn = sKNN(n_neighbors=n_neighbors)
        sknn.fit(XI, yI)
        h = 0.2
        x_min, x_max = XI[:, 0].min() - 1, XI[:, 0].max() + 1
        y_min, y_max = XI[:, 1].min() - 1, XI[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        Z = sknn.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)
    
    
        plt.subplot(1,2,2)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(XI[:, 0], XI[:, 1], c=yI, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Sklearn KNN 3-Class classification (k = %i)" % (n_neighbors))
        plt.show()


![png](http://www.bryantravissmith.com/img/GW04D1/output_7_0.png)



![png](http://www.bryantravissmith.com/img/GW04D1/output_7_1.png)


##Best K

We can use cross validation to try to find the best K to optimize for a metric.  Because the Iris data set has 3 labels, we can not use precision or recall.   We can use accuracy.   With a 20 Fold Cross Validation, we can estimate the accuracy for different K values.  


    from sklearn.metrics import accuracy_score,recall_score,precision_score
    from sklearn.cross_validation import KFold
    mean_accuracy = []
    mean_precision = []
    mean_recall = []
    k_range = range(1,200)
    for k in k_range:
        kf = KFold(len(yI), n_folds=20)
        knn = KNN(k,euclidean_distance)
        accuracy = []
        precision = []
        recall = []
        for train_index, test_index in kf:
            X_train, X_test = XI[train_index], XI[test_index]
            y_train, y_test = yI[train_index], yI[test_index]
            knn.fit(X_train,y_train)
            pred = knn.predict(X_test)
            accuracy.append(accuracy_score(y_test,pred))
            if len(np.unique(yI)) < 3:
                precision.append(precision_score(y_test,pred))
                recall.append(recall_score(y_test,pred))
        mean_accuracy.append(np.array(accuracy).mean())
        if len(np.unique(yI)) < 3:
            mean_precision.append(np.array(precision).mean())
            mean_recall.append(np.array(recall).mean())
    
    plt.figure(figsize=(10,5))
    plt.plot(k_range,mean_accuracy,label="Accuracy")
    if len(np.unique(yI)) < 3:
        plt.plot(k_range,mean_precision,label="Precision")
        plt.plot(k_range,mean_recall,label="Recall")
    plt.ylim([0.0,1.1])
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW04D1/output_9_0.png)


It seems like any k > 5 and k < 50 will do well this data set.  Really this should be done on a hold out set for a final estimate of the model.

##Recusion

We had an optional assignment to work on Recursive methods to preare for our afternoon spring involving decision trees.  We had to make trees and print trees using the following tree class.


    class TreeNode(object):
        def __init__(self, value, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right
            

The print method I developed was just to us a "|_" style to represent branches of the tree.


    def print_all(tree,level=1):
        response = "|_" + str(tree.value)
        if tree.left != None:
            response += '\n' +  " "*2*level +print_all(tree.left,level+1)
        if tree.right != None:
            response += "\n" +  " "*2*level +print_all(tree.right,level+1)
        return response
    
    Tree = TreeNode(10,TreeNode(5,TreeNode(4),TreeNode(5)),TreeNode(10))
    print print_all(Tree)

    |_10
      |_5
        |_4
        |_5
      |_10


We were asked to also find a method to find a value of a tree by summing.


    def sum_tree(root):
        if root.left == None:
            if root.right==None:
                return root.value
            else:
                return root.value+sum_tree(root.right)
        elif root.right == None:
            return root.value + sum_tree(root.left)
        else:
            return sum_tree(root.left) + sum_tree(root.right) + root.value
    
    sum_tree(Tree)




    34



We were also asked to come up with a pay to construct a tree that would give all possible out comes of n coin flips:


    def build_coinflip_tree(n,root=TreeNode("")):
        if n == 0:
            return root
        root.left = build_coinflip_tree(n-1,root=TreeNode("H"))
        root.right = build_coinflip_tree(n-1,root=TreeNode("T"))
        return root
    
    print print_all(build_coinflip_tree(3))

    |_
      |_H
        |_H
          |_H
          |_T
        |_T
          |_H
          |_T
      |_T
        |_H
          |_H
          |_T
        |_T
          |_H
          |_T


##Decision Trees

The afternoon paired spring involved creating and appending pythong classes to implement a Decision tree with pre and post purning.   We started with the simple golf dataset.


    from DecisionTree import DecisionTree
    import pandas as pd
    df = pd.read_csv('../data/playgolf.csv')
    df




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Windy</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sunny</td>
      <td>85</td>
      <td>85</td>
      <td>False</td>
      <td>Don't Play</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sunny</td>
      <td>80</td>
      <td>90</td>
      <td>True</td>
      <td>Don't Play</td>
    </tr>
    <tr>
      <th>2</th>
      <td>overcast</td>
      <td>83</td>
      <td>78</td>
      <td>False</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rain</td>
      <td>70</td>
      <td>96</td>
      <td>False</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rain</td>
      <td>68</td>
      <td>80</td>
      <td>False</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rain</td>
      <td>65</td>
      <td>70</td>
      <td>True</td>
      <td>Don't Play</td>
    </tr>
    <tr>
      <th>6</th>
      <td>overcast</td>
      <td>64</td>
      <td>65</td>
      <td>True</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sunny</td>
      <td>72</td>
      <td>95</td>
      <td>False</td>
      <td>Don't Play</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sunny</td>
      <td>69</td>
      <td>70</td>
      <td>False</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>9</th>
      <td>rain</td>
      <td>75</td>
      <td>80</td>
      <td>False</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sunny</td>
      <td>75</td>
      <td>70</td>
      <td>True</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>11</th>
      <td>overcast</td>
      <td>72</td>
      <td>90</td>
      <td>True</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>12</th>
      <td>overcast</td>
      <td>81</td>
      <td>75</td>
      <td>False</td>
      <td>Play</td>
    </tr>
    <tr>
      <th>13</th>
      <td>rain</td>
      <td>71</td>
      <td>80</td>
      <td>True</td>
      <td>Don't Play</td>
    </tr>
  </tbody>
</table>
</div>



First we will show the results of the Decision Tree we made:


    y = df.Result.values
    x = df.drop('Result',axis=1).values
    dt = DecisionTree()
    dt.fit(x, y,df.drop('Result',axis=1).columns)
    print dt

    Outlook
      |-> overcast:
      |     Play
      |-> no overcast:
      |     Temperature
      |     |-> < 80:
      |     |     Temperature
      |     |     |-> < 75:
      |     |     |     Temperature
      |     |     |     |-> < 71:
      |     |     |     |     Temperature
      |     |     |     |     |-> < 68:
      |     |     |     |     |     Don't Play
      |     |     |     |     |-> >= 68:
      |     |     |     |     |     Play
      |     |     |     |-> >= 71:
      |     |     |     |     Don't Play
      |     |     |-> >= 75:
      |     |     |     Play
      |     |-> >= 80:
      |     |     Don't Play


We can see the decision tree can split on the same variable multiple times, and we can change the spliting criteria.


    dt = DecisionTree(impurity_criterion='gini')
    dt.fit(x, y,df.drop('Result',axis=1).columns)
    print dt

    Outlook
      |-> overcast:
      |     Play
      |-> no overcast:
      |     Temperature
      |     |-> < 80:
      |     |     Temperature
      |     |     |-> < 68:
      |     |     |     Don't Play
      |     |     |-> >= 68:
      |     |     |     Temperature
      |     |     |     |-> < 71:
      |     |     |     |     Play
      |     |     |     |-> >= 71:
      |     |     |     |     Temperature
      |     |     |     |     |-> < 75:
      |     |     |     |     |     Don't Play
      |     |     |     |     |-> >= 75:
      |     |     |     |     |     Play
      |     |-> >= 80:
      |     |     Don't Play



    dt = DecisionTree(depth=2)
    dt.fit(x, y,df.drop('Result',axis=1).columns)
    print dt

    Outlook
      |-> overcast:
      |     Play
      |-> no overcast:
      |     Temperature
      |     |-> < 80:
      |     |     Play
      |     |-> >= 80:
      |     |     Don't Play



    dt = DecisionTree(leaf_size=7)
    dt.fit(x, y,df.drop('Result',axis=1).columns)
    print dt

    Outlook
      |-> overcast:
      |     Play
      |-> no overcast:
      |     Temperature
      |     |-> < 80:
      |     |     Temperature
      |     |     |-> < 75:
      |     |     |     Play
      |     |     |-> >= 75:
      |     |     |     Play
      |     |-> >= 80:
      |     |     Don't Play


##Post Pruning

We also implementing a post pruning procedure that takes a test set of data and labels, then prunes the tree to non decrease accuracy while reducing the size of the tree.


    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    iris = load_iris()
    y = iris.target
    x = iris.data
    X_trn,X_tst,y_trn,y_tst = train_test_split(x,y,test_size=.5)
    
    dt.fit(X_trn,y_trn)
    print dt
    print "Accuracy On Test Set:", accuracy_score(dt.predict(X_tst),y_tst)
    dt.prune(X_tst,y_tst)
    print dt
    print "Accuracy On Test Set:", accuracy_score(dt.predict(X_tst),y_tst)

    2
      |-> < 3.5:
      |     0
      |-> >= 3.5:
      |     3
      |     |-> < 1.8:
      |     |     2
      |     |     |-> < 5.6:
      |     |     |     1
      |     |     |-> >= 5.6:
      |     |     |     2
      |     |-> >= 1.8:
      |     |     2
      |     |     |-> < 4.9:
      |     |     |     1
      |     |     |-> >= 4.9:
      |     |     |     2
    Accuracy On Test Set: 0.906666666667
    2
      |-> < 3.5:
      |     0
      |-> >= 3.5:
      |     3
      |     |-> < 1.8:
      |     |     1
      |     |-> >= 1.8:
      |     |     2
    Accuracy On Test Set: 0.92


##Decision Tree Code

The decision tree class is built on a TreeNode class, similar to the recusion part of the lesson.   We altered the original to allow tracking of parent nodes for up and down tree transversals.


    from collections import Counter
    
    class TreeNode(object):
        '''
        A node class for a decision tree.
        '''
        def __init__(self):
            self.column = None  # (int)    index of feature to split on
            self.value = None  # value of the feature to split on
            self.categorical = True  # (bool) whether or not node is split on
                                     # categorial feature
            self.name = None    # (string) name of feature (or name of class in the
                                #          case of a list)
            self.parent = None
            self.left = None    # (TreeNode) left child
            self.right = None   # (TreeNode) right child
            self.leaf = False   # (bool)   true if node is a leaf, false otherwise
            self.classes = Counter()  # (Counter) only necessary for leaf node:
                                      #           key is class name and value is
                                      #           count of the count of data points
                                      #           that terminate at this leaf
            self.X_test = None
            self.y_test = None
    
        def predict_one(self, x):
            '''
            INPUT:
                - x: 1d numpy array (single data point)
            OUTPUT:
                - y: predicted label
    
            Return the predicted label for a single data point.
            '''
            if self.leaf:
                return self.name
            col_value = x[self.column]
    
            if self.categorical:
                if col_value == self.value:  ### REPLACE WITH YOUR CODE
                    return self.left.predict_one(x)
                else:
                    return self.right.predict_one(x)
            else:
                if col_value < self.value:  ### REPLACE WITH YOUR CODE
                    return self.left.predict_one(x)
                else:
                    return self.right.predict_one(x)
    
        def prune(self, X_test, y_test,parent=False):
            '''
            INPUT:
                - X_test: 2d np array
                - y_test: 1d np array
                - parent: Boolean
            OUTPUT:
    
            Prunes node if both children are leaves.  
            If not, call prune on children.
            If prune successful, call prune on parent node. 
            '''
            
            if not parent:
                self.X_test = X_test
                self.y_test = y_test
                
            if not self.leaf:
                if self.categorical:
                    mask = self.X_test[:,self.column]==self.value
                else:
                    mask = self.X_test[:,self.column] < self.value 
    
                if self.left.leaf and self.right.leaf:                
                    leftX = self.X_test[mask,:]
                    rightX = self.X_test[~mask,:]
                    lefty = self.y_test[mask]
                    righty = self.y_test[~mask]
                    if len(leftX) > 0:
                        left_y_pred = np.apply_along_axis(self.left.predict_one,1,leftX)
                    else:
                        left_y_pred = np.array([])
                    if len(rightX) > 0:
                        right_y_pred = np.apply_along_axis(self.right.predict_one,1,rightX)
                    else:
                        right_y_pred = np.array([])
                    accuracy = float(np.sum(left_y_pred==lefty)+np.sum(right_y_pred==righty))
    
                    new_counter = self.left.classes.copy()
                    for key in self.right.classes:
                        new_counter[key] += self.right.classes[key]
    
                    most_common = new_counter.most_common(1)[0][0]
    
                    node_acc = float(np.sum(self.y_test == most_common))
    
                    if node_acc >= accuracy:
                        self.left = None
                        self.right = None
                        self.leaf = True
                        self.classes = new_counter
                        self.name = most_common
                        self.parent.prune(None,None,parent=True)
                        
                else:
                    if self.left != None and not self.left.leaf:
                        self.left.prune(self.X_test[mask,:],self.y_test[mask])
                    if self.right != None and not self.right.leaf:
    
                        self.right.prune(self.X_test[~mask,:],self.y_test[~mask])
    
    
    
        # This is for visualizing your tree. You don't need to look into this code.
        def as_string(self, level=0, prefix=""):
            '''
            INPUT:
                - level: int (amount to indent)
            OUTPUT:
                - prefix: str (to start the line with)
    
            Return a string representation of the tree rooted at this node.
            '''
            result = ""
            if prefix:
                indent = "  |   " * (level - 1) + "  |-> "
                result += indent + prefix + "\n"
            indent = "  |   " * level
            result += indent + "  " + str(self.name) + "\n"
            if not self.leaf:
                if self.categorical:
                    left_key = str(self.value)
                    right_key = "no " + str(self.value)
                else:
                    left_key = "< " + str(self.value)
                    right_key = ">= " + str(self.value)
                result += self.left.as_string(level + 1, left_key + ":")
                result += self.right.as_string(level + 1, right_key + ":")
            return result
    
        def __repr__(self):
            return self.as_string().strip()



    import math
    
    class DecisionTree(object):
        '''
        A decision tree class.
        '''
    
        def __init__(self, leaf_size = None, depth = None, stop_percentage = None, error_threshold = None, impurity_criterion='entropy'):
            '''
            Initialize an empty DecisionTree.
            '''
    
            self.root = None  # root Node
            self.feature_names = None  # string names of features (for interpreting
                                       # the tree)
            self.categorical = None  # Boolean array of whether variable is
                                     # categorical (or continuous)
            self.impurity_criterion = self._entropy \
                                      if impurity_criterion == 'entropy' \
                                      else self._gini
            if type(leaf_size) == type(1) and leaf_size > 0:
                self.leaf_size = leaf_size
            else:
                self.leaf_size = 1
    
            if type(depth) == type(1) and depth > 0:
                self.depth = depth
            else:
                self.depth = 1e10
            if type(stop_percentage) == type(1.) and stop_percentage > 0:
                self.stop_percentage = stop_percentage
            else:
                self.stop_percentage = 1.
            if type(error_threshold) == type(1.) and error_threshold > 0:
                self.error_threshold = error_threshold
            else:
                self.error_threshold = 0
    
        def fit(self, X, y, feature_names=None):
            '''
            INPUT:
                - X: 2d numpy array
                - y: 1d numpy array
                - feature_names: numpy array of strings
            OUTPUT: None
    
            Build the decision tree.
            X is a 2 dimensional array with each column being a feature and each
            row a data point.
            y is a 1 dimensional array with each value being the corresponding
            label.
            feature_names is an optional list containing the names of each of the
            features.
            '''
    
            if feature_names is None or len(feature_names) != X.shape[1]:
                self.feature_names = np.arange(X.shape[1])
            else:
                self.feature_names = feature_names
    
            # Create True/False array of whether the variable is categorical
            is_categorical = lambda x: isinstance(x, str) or \
                                       isinstance(x, bool) or \
                                       isinstance(x, unicode)
            self.categorical = np.vectorize(is_categorical)(X[0])
    
            self.root = self._build_tree(X, y)
    
        def _build_tree(self, X, y,counter=0):
            '''
            INPUT:
                - X: 2d numpy array
                - y: 1d numpy array
            OUTPUT:
                - TreeNode
    
            Recursively build the decision tree. Return the root node.
            '''
    
            node = TreeNode()
            index, value, splits = self._choose_split_index(X, y)
            p = (np.unique(y, return_counts = True)[1].astype(float)/len(y)).max()
    
            if index is None or len(np.unique(y)) == 1 or counter >= self.depth or p > self.stop_percentage:
                node.leaf = True
                node.classes = Counter(y)
                node.name = node.classes.most_common(1)[0][0]
            else:
                X1, y1, X2, y2 = splits
                node.column = index
                node.name = self.feature_names[index]
                node.value = value
                node.categorical = self.categorical[index]
                
                node.left = self._build_tree(X1, y1,counter=(counter+1))
                node.left.parent = node
                
                node.right = self._build_tree(X2, y2,counter=(counter+1))
                node.right.parent = node
            return node
    
        def _entropy(self, y):
            '''
            INPUT:
                - y: 1d numpy array
            OUTPUT:
                - float
    
            Return the entropy of the array y.
            '''
            p = np.unique(y, return_counts = True)[1].astype(float)/len(y)
            return -1 * np.sum(p * np.log(p))
    
        def _gini(self, y):
            '''
            INPUT:
                - y: 1d numpy array
            OUTPUT:
                - float
    
            Return the gini impurity of the array y.
            '''
            p = np.unique(y, return_counts = True)[1].astype(float)/len(y)
            return 1 - np.sum(np.power(p,2))
    
        def _make_split(self, X, y, split_index, split_value):
            '''
            INPUT:
                - X: 2d numpy array
                - y: 1d numpy array
                - split_index: int (index of feature)
                - split_value: int/float/bool/str (value of feature)
            OUTPUT:
                - X1: 2d numpy array (feature matrix for subset 1)
                - y1: 1d numpy array (labels for subset 1)
                - X2: 2d numpy array (feature matrix for subset 2)
                - y2: 1d numpy array (labels for subset 2)
    
            Return the two subsets of the dataset achieved by the given feature and
            value to split on.
    
            Call the method like this:
            >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)
    
            X1, y1 is a subset of the data.
            X2, y2 is the other subset of the data.
            '''
            if self.categorical[split_index]:
                condition = X[:,split_index] == split_value
            else:
                condition = X[:,split_index] < split_value
            return X[condition == True,:], y[condition == True], X[condition == False,:], y[condition == False]
    
        def _information_gain(self, y, y1, y2):
            '''
            INPUT:
                - y: 1d numpy array
                - y1: 1d numpy array (labels for subset 1)
                - y2: 1d numpy array (labels for subset 2)
            OUTPUT:
                - float
    
            Return the information gain of making the given split.
    
            Use self.impurity_criterion(y) rather than calling _entropy or _gini
            directly.
            '''
            return self.impurity_criterion(y) - len(y1) * self.impurity_criterion(y1) / len(y) \
                    - len(y2) * self.impurity_criterion(y2) / len(y)
    
        def _choose_split_index(self, X, y):
            '''
            INPUT:
                - X: 2d numpy array
                - y: 1d numpy array
            OUTPUT:
                - index: int (index of feature)
                - value: int/float/bool/str (value of feature)
                - splits: (2d array, 1d array, 2d array, 1d array)
    
            Determine which feature and value to split on. Return the index and
            value of the optimal split along with the split of the dataset.
    
            Return None, None, None if there is no split which improves information
            gain.
    
            Call the method like this:
            >>> index, value, splits = self._choose_split_index(X, y)
            >>> X1, y1, X2, y2 = splits
            '''
            if self.leaf_size != None and len(y) <= self.leaf_size:
                return None, None, None
    
            index = -1
            value = -1
            max_gain = -1e10
            for i in xrange(len(X[0])):
                for split_value in np.unique(X[:, i]):
                    X1, y1, X2, y2 = self._make_split(X, y, i, split_value)
                    gain = self._information_gain(y, y1, y2)
                    if gain > max_gain:
                        max_gain = gain
                        value = split_value
                        index = i
    
            if max_gain > self.error_threshold:
                return index, value, self._make_split(X, y, index, value)
            else:
                return None, None, None
    
        def predict(self, X):
            '''
            INPUT:
                - X: 2d numpy array
            OUTPUT:
                - y: 1d numpy array
    
            Return an array of predictions for the feature matrix X.
            '''
    
            return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)
    
        def prune(self,X_test,y_test):
            self.root.prune(X_test,y_test)
    
        def __str__(self):
            '''
            Return string representation of the Decision Tree.
            '''
            return str(self.root)



    


    
