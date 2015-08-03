Title: Galvanize - Week 07 - Day 3
Date: 2015-07-15 10:20
Modified: 2015-07-15 10:30
Category: Galvanize
Tags: data-science, galvanize, map-reduce, mrjobs,generators
Slug: galvanize-data-science-07-03
Authors: Bryan Smith
Summary: Today we covered map reduce

#Galvanize Immersive Data Science

##Week 7 - Day 3

We continued our week of big data technology, with the goal of today getting use to functional programming for map-reduce jobs.   We are using the python package MRJob to run map-reduce jobs.

Our morning quiz had two questions/tasks

1.  Program a prime generator
2.  Write an acronym function that uses map, then reduce

For the first task I will use a function that checks if it is prime.   For small primes it is good enough.  Obviously it will take a long tome for very large prime numbers.


    import math
    def check_prime(number):
        if number == 1:
            return True
        limit = int(math.sqrt(number))+1
        for x in range(2,limit):
            if number % x == 0:
                return False
        return True
    
    def prime_generator():
        i = 1
        while True:
            if check_prime(i):
                yield i
            i+= 1
    
    pg = prime_generator()
    
    print "First 25 Primes: ", [pg.next() for i in range(25)]

    First 25 Primes:  [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]



    def acronym(string):
        first_letters = map(lambda x: x[0],string.upper().split())
        acronym = reduce(lambda x,y: x+y, first_letters)
        return acronym
    
    acronym("Map reduce is awesome!")




    'MRIA'



##MRJob

###News Groups
MRJob is a python package for map-reduce that can be run local for on ec2 or hdsf instance.  For now we are going to run them locally on a subset of the news group data.   We are going to subset groups to 

1. comp.windows.x
2. rec.motorcycles
3. sci.med

And only have a fraction of the posts from each group.   This selection is arbitrary, as is the subset.   Its is only to make it so our code development can iterate faster becasue we are running it on less data.  

Our first task we are ask to complete is to get the word count by news group.  Becuase the newsgroup data is in a directory structure, we will use the os variable 'map_reduce_file', and get the second to the last value to label the topic.  

Our newest instructor is from the Data Engineering program, and he is also teaching us some bash commands, and how to use them in python.


    %%writefile code/wordcounts_bytopic.py
    from mrjob.job import MRJob
    from string import punctuation
    import os
    
    class MRWordFreqCount(MRJob):
        
        def mapper(self, _, line):
            topic = os.environ['map_input_file'].split("/")
            topic_str = topic[-2]
            for word in line.split():
                yield (topic_str+"_"+word.strip(punctuation).lower(), 1)
    
        def reducer(self, word, counts):
            yield (word, sum(counts))
    
    if __name__ == '__main__':
        import sys
        from StringIO import StringIO
        #Suppress the error outputs from not setting up a local config file for MRJob
        #sys.stdout = StringIO();
        sys.stderr = StringIO();
        MRWordFreqCount().run()
        #sys.stdout = sys.__stdout__;
        sys.stderr = sys.__stderr__;

    Overwriting code/wordcounts_bytopic.py



    !python code/wordcounts_bytopic.py mini_newsgroups/ > output/wordcounts_by_topic.txt
    !awk 'NR >= 800 && NR <= 810' output/wordcounts_by_topic.txt

    Traceback (most recent call last):
      File "code/wordcounts_bytopic.py", line 17, in <module>
        suppress(MRWordFreqCount().run())
    NameError: name 'suppress' is not defined


We can look later in the file and see the other topics


    !awk 'NR >= 10800 && NR <= 10810' output/wordcounts_by_topic.txt

    "rec.motorcycles_threads"	1
    "rec.motorcycles_three"	7
    "rec.motorcycles_three-notch"	1
    "rec.motorcycles_threw"	1
    "rec.motorcycles_thrive"	1
    "rec.motorcycles_throgmorton"	1
    "rec.motorcycles_throttle"	1
    "rec.motorcycles_through"	8
    "rec.motorcycles_through-studs"	2
    "rec.motorcycles_throughout"	1
    "rec.motorcycles_throwing"	1


###New York Times

We havea json file with apprximately 1400 new york times articles and their meta data.  We want to construct a map-reduce job by article


    %%writefile code/articles_wordcount.py
    from mrjob.job import MRJob
    from mrjob.step import MRStep
    import json
    import re
    import string
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.porter import PorterStemmer
    tokenizer = RegexpTokenizer('\w+')
    WORD_RE = re.compile(r"^[\D]+$")
    stemmer = PorterStemmer()
    
    class MRMostUsedWord(MRJob):
        
        def steps(self):
            return [
                MRStep(mapper=self.mapper_get_words,
                       combiner=self.combiner_count_words,
                       reducer=self.reducer_count_words)
            ]
        
        def mapper_get_words(self, _, line):
            data = json.loads(line)
            content = "\n".join(data['content']).encode('ascii','ignore')
            for word in tokenizer.tokenize(content):
                yield (stemmer.stem(word.lower()), (1,data['_id']))
                
        def combiner_count_words(self, word, counts):
            # optimization: sum the words we've seen so far
            temp = [(value,id) for value,id in counts]
            total = sum([value for value, id in temp])
            for value,id in temp:
                yield ((id,word,total), value)
    
        def reducer_count_words(self, (id,word,total), counts):
            # send all (num_occurrences, word) pairs to the same reducer.
            # num_occurrences is so we can easily use Python's max() function.
            yield ((word,id),(sum(counts),total))
    
    if __name__ == '__main__':
        import sys
        from StringIO import StringIO
        #Suppress the error outputs from not setting up a local config file for MRJob
        #sys.stdout = StringIO();
        sys.stderr = StringIO();
        MRMostUsedWord.run()
        #sys.stdout = sys.__stdout__;
        sys.stderr = sys.__stderr__;

    Overwriting code/articles_wordcount.py



    !python code/articles_wordcount.py data/articles.json > output/articles_wordcount.txt
    !awk 'NR >= 800 && NR <= 810' output/articles_wordcount.txt

    ["put", "5233249838f0d8062fddf6a0"]	[1, 307]
    ["rare", "5233249838f0d8062fddf6a0"]	[1, 84]
    ["ration", "5233249838f0d8062fddf6a0"]	[1, 10]
    ["raymond", "5233249838f0d8062fddf6a0"]	[2, 19]
    ["read", "5233249838f0d8062fddf6a0"]	[1, 193]
    ["reason", "5233249838f0d8062fddf6a0"]	[1, 136]
    ["recent", "5233249838f0d8062fddf6a0"]	[2, 414]
    ["red", "5233249838f0d8062fddf6a0"]	[1, 206]
    ["region", "5233249838f0d8062fddf6a0"]	[1, 97]
    ["relat", "5233249838f0d8062fddf6a0"]	[1, 103]
    ["remain", "5233249838f0d8062fddf6a0"]	[1, 237]


## Triadic Closure

We were told that LinkedIn origianlly recommended relationships through the process of triadic closures.  If I have two collegues, Joe and Jill, then if Joe and Jill become collegues, that would close the triad.  We are going to be using the map-reduce jobs to engage in this process using the facebook social graph [data](http://snap.stanford.edu/data/egonets-Facebook.html).

To ease development we will first a very small subset of the data, then run it on the full dataset.


    !head -10 data/edges.txt

    0 1
    0 2
    0 3
    0 4
    0 5
    0 6
    0 7
    0 8
    0 9
    0 10


The data is a collection of edges.  Friend 0 is friends with friend 1.  The reverse relationship is not listed, so we need to make sure to keep track of this undirectedness as we do produce the job.  Our first task is to get a list of friends for each user.

We will do this by making each pair of friendships.  For example, (0,1) becomes (0,1) and (1,0).  We then use the first value as a key, and get all the friends (values) associated with that key.  That will give use a list of all the friends of a user. 


    %%writefile code/getfriends.py
    from mrjob.job import MRJob
    from string import punctuation
    import os
    
    class MRGetFriends(MRJob):
    
        def mapper(self, _, line):
            f1, f2 = line.split()
            pairs = [(f1,f2),(f2,f1)]
            for pair in pairs:
                yield int(pair[0]),int(pair[1])
            
        def reducer(self,key,values):
            friends = [val for val in values]
            yield key, friends
    
    if __name__ == '__main__':
        import sys
        from StringIO import StringIO
        #Suppress the error outputs from not setting up a local config file for MRJob
        sys.stderr = StringIO()
        MRGetFriends.run()
        sys.stderr = sys.__stderr__
        
        

    Overwriting code/getfriends.py



    !python code/getfriends.py data/mini_edges.txt > output/get_friends_mini.txt
    !cat output/get_friends_mini.txt

    0	[1, 2, 5]
    1	[0, 3, 4]
    2	[0, 3, 4]
    3	[1, 2, 4]
    4	[1, 2, 3, 5]
    5	[0, 4]


The next step is to get the number of mutual friends each pairs of users have, but we need to remove users that are already friends.  To do this we can take the previous lists and make pairs.

Using the first row we do the following:

0 [1,2,5]  => ((0,1),0),((0,2),0),((0,5),0) plus ((1,2),1),((1,5),1),((2,5),1)

This has a list of relationships with 1 is for evidence of having a friend in common, and zero for no evidence for having a friend in commom (even though they are friends).

Collect all the keys together and see if the values add them up.  If the total is less then the number of values, then we know they have to already be friends.   Otherwise, we know they are not friends and know how my friends they have in common.


    %%writefile code/getfriends2.py
    from mrjob.job import MRJob
    from mrjob.step import MRStep
    from string import punctuation
    from itertools import combinations
    
    class MRGetFriends(MRJob):
        def steps(self):
            return [
                MRStep(mapper=self.mapper1,reducer=self.reducer1),
                MRStep(mapper=self.mapper2,reducer=self.reducer2)
            ]
        def mapper1(self, _, line):
            f1, f2 = line.split()
            pairs = [(f1,f2),(f2,f1)]
            for pair in pairs:
                yield int(pair[0]),int(pair[1])
            
        def reducer1(self,key,values):
            friends = [val for val in values]
            yield key, friends
            
        def mapper2(self, key, values):
            results = []
            vals = [val for val in values]
            for val in vals:
                results.append((tuple(sorted([key,val])),0))
                
            combination = combinations(vals,2)
            for c in combination:
                results.append((sorted(c),1))
                
            for result in results:
                yield result
        
        def reducer2(self, key, values):
            total = 0
            count = 0
            for val in values:
                total += val
                count += 1
                
            if count==total:
                yield key, total
            
    if __name__ == '__main__':
        import sys
        from StringIO import StringIO
        #Suppress the error outputs from not setting up a local config file for MRJob
        sys.stderr = StringIO()
        MRGetFriends.run()
        sys.stderr = sys.__stderr__

    Overwriting code/getfriends2.py



    !python code/getfriends2.py data/mini_edges.txt > output/get_friends_mini2.txt
    !cat output/get_friends_mini2.txt

    [0, 3]	2
    [0, 4]	3
    [1, 2]	3
    [1, 5]	2
    [2, 5]	2
    [3, 5]	1


Now that we have a list of pairs that are not friends, but have friends in common, we can generate suggestions for each user by finding the other person they have the most friends in common with.  This, however, is an asymetric suggestion.  Just because user 1 is suggested user 5 does not mean that user 5 will be suggested user 1.  

To generate this list we will take the above output and map them to a pair of relationships and counts.

$$[ \ 0, \ 3] \ 2 \ \implies ( \ 0, \ ( \ 2, \ 3) \ ), ( \ 3, \ ( \ 2, \ 0) \ )$$

We then go by key and select the tuple with the highest friend count, and return the corresponding userid. 


    %%writefile code/getfriends3.py
    from mrjob.job import MRJob
    from mrjob.step import MRStep
    from string import punctuation
    from itertools import combinations
    
    class MRGetFriends(MRJob):
        def steps(self):
            return [
                MRStep(mapper=self.mapper1,reducer=self.reducer1),
                MRStep(mapper=self.mapper2,reducer=self.reducer2),
                MRStep(mapper=self.mapper3,reducer=self.reducer3)
            ]
        def mapper1(self, _, line):
            f1, f2 = line.split()
            pairs = [(f1,f2),(f2,f1)]
            for pair in pairs:
                yield int(pair[0]),int(pair[1])
            
        def reducer1(self,key,values):
            friends = [val for val in values]
            yield key, friends
            
        def mapper2(self, key, values):
            results = []
            vals = [val for val in values]
            for val in vals:
                results.append((tuple(sorted([key,val])),0))
                
            combination = combinations(vals,2)
            for c in combination:
                results.append((sorted(c),1))
                
            for result in results:
                yield result
        
        def reducer2(self, key, values):
            total = 0
            count = 0
            for val in values:
                total += val
                count += 1
                
            if count==total:
                yield key, total
        
        def mapper3(self, key, value):
            keys = [(key[0],key[1]),(key[1],key[0])]
            for k in keys:
                yield (k[0], (value,k[1]))
            
        def reducer3(self,key,values):
            max_value = max(values)
            yield key, max_value[1]
            
    if __name__ == '__main__':
        import sys
        from StringIO import StringIO
        #Suppress the error outputs from not setting up a local config file for MRJob
        sys.stderr = StringIO()
        MRGetFriends.run()
        sys.stderr = sys.__stderr__

    Overwriting code/getfriends3.py



    !python code/getfriends3.py data/mini_edges.txt > output/get_friends_mini4.txt
    !cat output/get_friends_mini4.txt

    0	4
    1	2
    2	1
    3	0
    4	0
    5	2


###Full dataset.   

Now that we have a full map-reduce job built, we can run it on the full dataset. 



    !python code/getfriends3.py data/edges.txt > output/get_friends_full3.txt


    !head -20 output/get_friends_full3.txt

    0	348
    1	302
    10	271
    100	171
    1000	1366
    1001	1320
    1002	1627
    1003	1584
    1004	1126
    1005	1472
    1006	1589
    1007	1893
    1008	1863
    1009	1479
    101	196
    1010	1236
    1011	1428
    1012	1645
    1013	1778
    1014	1591


##AWS

We were also instructed to use the mrjob interaction with AWS.  Amazon has changed their API, and this is not currently working.   I saw on github that they are working on this, but the current changes are wating for code review before being released.  



    
