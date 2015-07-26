
#Galvanize Immersive Data Science

##Week 7 - Day 4

Today we stared our two day Spark tutorials, so our mini quiz was downloading and running a stand alone version of spark.  We went to the [Spark download page](http://spark.apache.org/downloads.html) and downloaded the "Prebuilt for Hadoop X.X" gzip file.   After it was downloaded and uncompressed, we altered our .bash_profile point $SPARK_HOME to the new directory, and extend the python path to include the python directory in this spark folder.   I am on the mac, so I addded these two lines into my .bash_profile.

>export SPARK_HOME=/Users/bryantravissmith/spark-1.4.1-bin-hadoop1  
>export PYTHONPATH=/Users/bryantravissmith/spark-1.4.1-bin-hadoop1/python/:$PYTHONPATH

##Starting a Local Cluster

Spark has a master node and worker nodes.  The following commands will start the system on the mac.

###Master

>  ${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.master.Master -h 127.0.0.1 -p 7077 --webui-port 8080

###Worker

Use this once for each worker you wish to added to the master.

> ${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.worker.Worker -c 1 -m 1G spark://127.0.0.1:7077

We used [`tmux`](http://tmux.sourceforge.net/) to run these commends in a connected terminal, then opened an ipython notebook.  The commands looked like this.

>tmux new -s master    
>\${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.master.Master -h 127.0.0.1 -p 7077 --webui-port 8080   
>ctrl+b, d  
>tmux new -s worker1   
>\${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.worker.Worker -c 1 -m 1G spark://127.0.0.1:7077   
>ctrl+b, d  
>tmux new -s worker2    
>\${SPARK_HOME}/bin/spark-class org.apache.spark.deploy.worker.Worker -c 1 -m 1G spark://127.0.0.1:7077  
>ctrl+b, d  
>  IPYTHON_OPTS="notebook"  \${SPARK_HOME}/bin/pyspark --master spark://127.0.0.1:7077 --executor-memory 1G --driver-memory 1G

All my postes are initially writen in an iPython notebook, and is how I am currently running this session.

##Airline Delays

We are going to download airline flight information and explore an analysis using spark.   This is going to be different than previous work we have done because all the work is structured in the map-reduce framework.  We are also getting in the habit of loading data from hosted services, instead of working with local data files.  

This data had arrival delays at the destination coded as 'ARR_DELAY', and the departire delay heading to the destination coded as 'DEP_DELAY'.  These values are in minutes.   If the flight is cancled, we will say the delay is 300 minutes (5 hours).


    link = 's3n://AKIAJRH3ZAWFYUFN5WIA:xSztE4PvK7GQ3zHuohSoOgrOdV9OosfP+4WXod0R@mortar-example-data/airline-data'
    airline = sc.textFile(link)


    airline.count()




    5113194



The airline is an RDD structure.  It is not data, but instructions to go through the data.  Each time we do this it will cycle through 5 million rows in the data.  To make analysis faster, we will build it out on a subset of the data, and run the final analysis on the full dataset.


    mini_airline = sc.parallelize(airline.take(1000))


    mini_airline.take(10)




    [u'"YEAR","MONTH","UNIQUE_CARRIER","ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","DEP_DELAY","DEP_DELAY_NEW","ARR_DELAY","ARR_DELAY_NEW","CANCELLED",',
     u'2012,4,"AA",12478,12892,-4.00,0.00,-21.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-7.00,0.00,-65.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-6.00,0.00,-63.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-6.00,0.00,5.00,5.00,0.00,',
     u'2012,4,"AA",12478,12892,-2.00,0.00,-39.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-6.00,0.00,-34.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-8.00,0.00,-16.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-7.00,0.00,-19.00,0.00,0.00,',
     u'2012,4,"AA",12478,12892,-9.00,0.00,-2.00,0.00,0.00,']



We see that data structure is year, month, airline, 2x airport ideas, then the time information in minutes.   The first 9 data points do not look like they have the 'cancelled' variables listed.  The goal is to group this data to find the top 10 worst and best airports in terms of arrival and departure delays.  

First we will get the data we want, id's and delays, then we will combind them by airport in terms of arrival and departures.   After that we will sort the data, determine the best airports.


    def make_row(x):
        values = x.split(",")
        dictionary = {'DEST_AIRPORT_ID':values[4], 'ORIGIN_AIRPORT_ID':values[3]}
        
        if values[-1] != '0.00':
            dictionary['DEP_DELAY'] = -float(5*60)
        elif values[6] == '': 
            dictionary['DEP_DELAY'] = float(0)
        else:
            dictionary['DEP_DELAY'] = float(values[6])
        
        if values[7] == '': 
            dictionary['ARR_DELAY'] = -1*dictionary['DEP_DELAY']
        else:
            dictionary['ARR_DELAY'] = (float(values[7])-dictionary['DEP_DELAY'])
    
        return dictionary
    
    mini_airline.map(lambda x: x[:-1]) \
                .filter(lambda x: not x.startswith('"YEAR"')) \
                .map(make_row) \
                .take(10)




    [{'ARR_DELAY': -21.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -65.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -63.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': 5.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -39.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -34.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -16.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -19.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -2.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'},
     {'ARR_DELAY': -17.0,
      'DEP_DELAY': 0.0,
      'DEST_AIRPORT_ID': u'12892',
      'ORIGIN_AIRPORT_ID': u'12478'}]



I will continue to test with the mini-dataset.   Since RDDs are instructions, and not actually loaded, I am going to construct the total/final RDD as I move along this process.


    all_airline = airline.map(lambda x: x[:-1]) \
                .filter(lambda x: not x.startswith('"YEAR"')) \
                .map(make_row)


  
7. Instead of dictionaries, make 2 RDDs where the items are tuples.
   The first RDD will contain tuples `(DEST_AIRPORT_ID, ARR_DELAY)`. 
   The other RDD will contain `(ORIGIN_AIRPORT_ID, DEP_DELAY)`.
   Run a `.first()` or `.take()` to confirm your results.


    arr_rdd = mini_airline.map(lambda x: x[:-1]) \
                .filter(lambda x: not x.startswith('"YEAR"')) \
                .map(make_row) \
                .map(lambda x: (x['DEST_AIRPORT_ID'], x['ARR_DELAY']))
    print arr_rdd.take(5)
    
    dep_rdd = mini_airline.map(lambda x: x[:-1]) \
                .filter(lambda x: not x.startswith('"YEAR"')) \
                .map(make_row) \
                .map(lambda x: (x['ORIGIN_AIRPORT_ID'], x['DEP_DELAY']))
    print dep_rdd.take(5)

    [(u'12892', -21.0), (u'12892', -65.0), (u'12892', -63.0), (u'12892', 5.0), (u'12892', -39.0)]
    [(u'12478', 0.0), (u'12478', 0.0), (u'12478', 0.0), (u'12478', 0.0), (u'12478', 0.0)]



    arr_all = all_airline.map(lambda x: (x['DEST_AIRPORT_ID'], x['ARR_DELAY']))
                
    dep_all = all_airline.map(lambda x: (x['ORIGIN_AIRPORT_ID'], x['DEP_DELAY']))

We have to RDD's, one for departures and one for arrivals.   In order to do the map reduce average, we need to add a new value of 1 to the values.  We can then reduce by each airport id (key), and add the delays to the 1, and get the total minutes delayed and the total number of flights.   In the last step we can divide these numbers to get the average delay.


    arr_rdd.map(lambda x: (x[0],(x[1],1))) \
        .reduceByKey(lambda (t1,c1),(t2,c2): (t1+t2,c1+c2)) \
        .map(lambda (k,(t,c)): (k,float(t)/c)) \
        .collect()




    [(u'11298', -3.7),
     (u'13830', -7.8),
     (u'12264', -4.266666666666667),
     (u'11618', -5.066666666666666),
     (u'12478', -14.086842105263157),
     (u'12892', -17.50574712643678),
     (u'12173', -5.883333333333334),
     (u'14771', -6.011363636363637)]




    dep_rdd.map(lambda x: (x[0],(x[1],1))) \
        .reduceByKey(lambda (t1,c1),(t2,c2): (t1+t2,c1+c2)) \
        .map(lambda (k,(t,c)): (k,float(t)/c)) \
        .collect()




    [(u'10721', 2.7333333333333334),
     (u'11298', 15.616666666666667),
     (u'13830', 10.183333333333334),
     (u'12264', 7.633333333333334),
     (u'12478', 6.042471042471043),
     (u'12892', 6.471590909090909),
     (u'12173', 2.716666666666667),
     (u'14771', 6.263513513513513)]




    full_arr_all = arr_all.map(lambda x: (x[0],(x[1],1))) \
        .reduceByKey(lambda (t1,c1),(t2,c2): (t1+t2,c1+c2)) \
        .map(lambda (k,(t,c)): (k,float(t)/c)) 
        
    full_dep_all = dep_all.map(lambda x: (x[0],(x[1],1))) \
        .reduceByKey(lambda (t1,c1),(t2,c2): (t1+t2,c1+c2)) \
        .map(lambda (k,(t,c)): (k,float(t)/c)) 

We are going to use these RDD's a couple of times for computations.   There is a persist function that allows us to store the data in memory to make repeated calculations quicker.   


    full_arr_all.setName("full_arr_all").persist()
    full_dep_all.setName("full_dep_all").persist()




    full_dep_all PythonRDD[88] at RDD at PythonRDD.scala:43




    #Least Arrival Delays
    full_arr_all.sortBy(lambda (k,v):v, ascending=False).take(10)




    [(u'13388', 38.46979865771812),
     (u'13424', 29.59053685168335),
     (u'10170', 26.305357142857144),
     (u'10930', 15.864705882352942),
     (u'14487', 15.296669248644461),
     (u'10551', 15.04993909866017),
     (u'10157', 13.659121621621622),
     (u'11002', 12.230910763569458),
     (u'10165', 10.727272727272727),
     (u'12177', 10.399602385685885)]




    #Longest Arrival Delays
    full_arr_all.sortBy(lambda (k,v):v, ascending=True).take(10)




    [(u'12343', -12.16822429906542),
     (u'13541', -11.927272727272728),
     (u'11415', -10.93968253968254),
     (u'10158', -10.602739726027398),
     (u'11537', -10.430456852791878),
     (u'11252', -10.345300261096606),
     (u'11111', -10.33978494623656),
     (u'10466', -9.875),
     (u'10154', -8.968454258675079),
     (u'12007', -8.896758703481392)]




    #Smallest Departure Delays
    full_dep_all.sortBy(lambda (k,v):v, ascending=False).take(10)




    [(u'13541', 35.91818181818182),
     (u'10154', 21.757097791798106),
     (u'15356', 20.4),
     (u'12016', 17.964401294498384),
     (u'15295', 17.759493670886076),
     (u'14512', 16.536585365853657),
     (u'10728', 15.7),
     (u'15048', 14.919597989949748),
     (u'10208', 10.619544945915703),
     (u'11641', 10.550997365449755)]




    #Longest Departure Delays
    full_dep_all.sortBy(lambda (k,v):v, ascending=True).take(10)




    [(u'13388', -31.391304347826086),
     (u'10170', -23.6875),
     (u'15041', -17.313357400722023),
     (u'12888', -16.22641509433962),
     (u'14905', -11.808695652173913),
     (u'12335', -11.481132075471699),
     (u'10551', -10.350791717417783),
     (u'11274', -10.25),
     (u'13424', -8.808007279344858),
     (u'10779', -8.094915254237288)]



##MLlib 

The afternoon assignment has to do with doing machine learning in spark.   We are given some news data, and our goal is to build a NaiveBayes model that predicts which news group an given post belongs in.   


    import string
    import json 
    import pickle as pkl
    from pyspark.mllib.feature import HashingTF
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes
    from collections import Counter
    
    data_raw = sc.textFile('s3n://AKIAJRH3ZAWFYUFN5WIA:xSztE4PvK7GQ3zHuohSoOgrOdV9OosfP+4WXod0R@sparkdatasets/news.txt')
    data_raw.count()




    13087




    


    data_raw.repartition(4)




    MapPartitionsRDD[123] at repartition at NativeMethodAccessorImpl.java:-2




    data_raw.take(2)




    [u'{"text": "From: twillis@ec.ecn.purdue.edu (Thomas E Willis)\\nSubject: PB questions...\\nOrganization: Purdue University Engineering Computer Network\\nDistribution: usa\\nLines: 36\\n\\nwell folks, my mac plus finally gave up the ghost this weekend after\\nstarting life as a 512k way back in 1985.  sooo, i\'m in the market for a\\nnew machine a bit sooner than i intended to be...\\n\\ni\'m looking into picking up a powerbook 160 or maybe 180 and have a bunch\\nof questions that (hopefully) somebody can answer:\\n\\n* does anybody know any dirt on when the next round of powerbook\\nintroductions are expected?  i\'d heard the 185c was supposed to make an\\nappearence \\"this summer\\" but haven\'t heard anymore on it - and since i\\ndon\'t have access to macleak, i was wondering if anybody out there had\\nmore info...\\n\\n* has anybody heard rumors about price drops to the powerbook line like the\\nones the duo\'s just went through recently?\\n\\n* what\'s the impression of the display on the 180?  i could probably swing\\na 180 if i got the 80Mb disk rather than the 120, but i don\'t really have\\na feel for how much \\"better\\" the display is (yea, it looks great in the\\nstore, but is that all \\"wow\\" or is it really that good?).  could i solicit\\nsome opinions of people who use the 160 and 180 day-to-day on if its worth\\ntaking the disk size and money hit to get the active display?  (i realize\\nthis is a real subjective question, but i\'ve only played around with the\\nmachines in a computer store breifly and figured the opinions of somebody\\nwho actually uses the machine daily might prove helpful).\\n\\n* how well does hellcats perform?  ;)\\n\\nthanks a bunch in advance for any info - if you could email, i\'ll post a\\nsummary (news reading time is at a premium with finals just around the\\ncorner... :( )\\n--\\nTom Willis  \\\\  twillis@ecn.purdue.edu    \\\\    Purdue Electrical Engineering\\n---------------------------------------------------------------------------\\n\\"Convictions are more dangerous enemies of truth than lies.\\"  - F. W.\\nNietzsche\\n", "label_name": "comp.sys.mac.hardware", "label": 4}',
     u'{"text": "From: jgreen@amber (Joe Green)\\nSubject: Re: Weitek P9000 ?\\nOrganization: Harris Computer Systems Division\\nLines: 14\\nDistribution: world\\nNNTP-Posting-Host: amber.ssd.csd.harris.com\\nX-Newsreader: TIN [version 1.1 PL9]\\n\\nRobert J.C. Kyanko (rob@rjck.UUCP) wrote:\\n> abraxis@iastate.edu writes in article <abraxis.734340159@class1.iastate.edu>:\\n> > Anyone know about the Weitek P9000 graphics chip?\\n> As far as the low-level stuff goes, it looks pretty nice.  It\'s got this\\n> quadrilateral fill command that requires just the four points.\\n\\nDo you have Weitek\'s address/phone number?  I\'d like to get some information\\nabout this chip.\\n\\n--\\nJoe Green\\t\\t\\t\\tHarris Corporation\\njgreen@csd.harris.com\\t\\t\\tComputer Systems Division\\n\\"The only thing that really scares me is a person with no sense of humor.\\"\\n\\t\\t\\t\\t\\t\\t-- Jonathan Winters\\n", "label_name": "comp.graphics", "label": 1}']



**We can see that each line is a json object, so we will need to use need to map each line to a dictionary.**


    data_json = data_raw.map(lambda x: json.loads(x))
    data_json.cache()
    data_json.take(2)




    [{u'label': 4,
      u'label_name': u'comp.sys.mac.hardware',
      u'text': u'From: twillis@ec.ecn.purdue.edu (Thomas E Willis)\nSubject: PB questions...\nOrganization: Purdue University Engineering Computer Network\nDistribution: usa\nLines: 36\n\nwell folks, my mac plus finally gave up the ghost this weekend after\nstarting life as a 512k way back in 1985.  sooo, i\'m in the market for a\nnew machine a bit sooner than i intended to be...\n\ni\'m looking into picking up a powerbook 160 or maybe 180 and have a bunch\nof questions that (hopefully) somebody can answer:\n\n* does anybody know any dirt on when the next round of powerbook\nintroductions are expected?  i\'d heard the 185c was supposed to make an\nappearence "this summer" but haven\'t heard anymore on it - and since i\ndon\'t have access to macleak, i was wondering if anybody out there had\nmore info...\n\n* has anybody heard rumors about price drops to the powerbook line like the\nones the duo\'s just went through recently?\n\n* what\'s the impression of the display on the 180?  i could probably swing\na 180 if i got the 80Mb disk rather than the 120, but i don\'t really have\na feel for how much "better" the display is (yea, it looks great in the\nstore, but is that all "wow" or is it really that good?).  could i solicit\nsome opinions of people who use the 160 and 180 day-to-day on if its worth\ntaking the disk size and money hit to get the active display?  (i realize\nthis is a real subjective question, but i\'ve only played around with the\nmachines in a computer store breifly and figured the opinions of somebody\nwho actually uses the machine daily might prove helpful).\n\n* how well does hellcats perform?  ;)\n\nthanks a bunch in advance for any info - if you could email, i\'ll post a\nsummary (news reading time is at a premium with finals just around the\ncorner... :( )\n--\nTom Willis  \\  twillis@ecn.purdue.edu    \\    Purdue Electrical Engineering\n---------------------------------------------------------------------------\n"Convictions are more dangerous enemies of truth than lies."  - F. W.\nNietzsche\n'},
     {u'label': 1,
      u'label_name': u'comp.graphics',
      u'text': u'From: jgreen@amber (Joe Green)\nSubject: Re: Weitek P9000 ?\nOrganization: Harris Computer Systems Division\nLines: 14\nDistribution: world\nNNTP-Posting-Host: amber.ssd.csd.harris.com\nX-Newsreader: TIN [version 1.1 PL9]\n\nRobert J.C. Kyanko (rob@rjck.UUCP) wrote:\n> abraxis@iastate.edu writes in article <abraxis.734340159@class1.iastate.edu>:\n> > Anyone know about the Weitek P9000 graphics chip?\n> As far as the low-level stuff goes, it looks pretty nice.  It\'s got this\n> quadrilateral fill command that requires just the four points.\n\nDo you have Weitek\'s address/phone number?  I\'d like to get some information\nabout this chip.\n\n--\nJoe Green\t\t\t\tHarris Corporation\njgreen@csd.harris.com\t\t\tComputer Systems Division\n"The only thing that really scares me is a person with no sense of humor."\n\t\t\t\t\t\t-- Jonathan Winters\n'}]



Great - we have the data in the correct format to easily manipulate.   We are going to quickly make a dictionary between the label_name and the label for the dataset.   Then we are going to start the text processing.


    labels = data_json.map(lambda x: (x['label'],x['label_name'])).distinct().collect()
    labels = {key:value for key,value in labels}
    print labels

    {0: u'alt.atheism', 1: u'comp.graphics', 2: u'comp.os.ms-windows.misc', 3: u'comp.sys.ibm.pc.hardware', 4: u'comp.sys.mac.hardware', 5: u'comp.windows.x', 6: u'misc.forsale', 7: u'rec.autos', 8: u'rec.motorcycles', 9: u'rec.sport.baseball', 10: u'rec.sport.hockey', 11: u'sci.crypt', 12: u'sci.electronics', 13: u'sci.med', 14: u'sci.space', 15: u'soc.religion.christian', 16: u'talk.politics.guns', 17: u'talk.politics.mideast', 18: u'talk.politics.misc', 19: u'talk.religion.misc'}



    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    
    stop = stopwords.words('english')
    tokenizer = RegexpTokenizer('\w+')
    stemmer = PorterStemmer()
    
    def token_text(text):
        tokens = []
        content = text.encode('ascii','ignore')
        for word in tokenizer.tokenize(content):
            word = word.lower()
            if word not in stop and len(word) > 2:
                tokens.append(stemmer.stem(word).encode('ascii'))
        return tokens
    
    print data_json.map(lambda x: (x['label'],token_text(x['text']))).take(1)

    [(4, ['twilli', 'ecn', 'purdu', 'edu', 'thoma', 'willi', 'subject', 'question', 'organ', 'purdu', 'univers', 'engin', 'comput', 'network', 'distribut', 'usa', 'line', 'well', 'folk', 'mac', 'plu', 'final', 'gave', 'ghost', 'weekend', 'start', 'life', '512k', 'way', 'back', '1985', 'sooo', 'market', 'new', 'machin', 'bit', 'sooner', 'intend', 'look', 'pick', 'powerbook', '160', 'mayb', '180', 'bunch', 'question', 'hope', 'somebodi', 'answer', 'anybodi', 'know', 'dirt', 'next', 'round', 'powerbook', 'introduct', 'expect', 'heard', '185c', 'suppos', 'make', 'appear', 'summer', 'haven', 'heard', 'anymor', 'sinc', 'access', 'macleak', 'wonder', 'anybodi', 'info', 'anybodi', 'heard', 'rumor', 'price', 'drop', 'powerbook', 'line', 'like', 'one', 'duo', 'went', 'recent', 'impress', 'display', '180', 'could', 'probabl', 'swing', '180', 'got', '80mb', 'disk', 'rather', '120', 'realli', 'feel', 'much', 'better', 'display', 'yea', 'look', 'great', 'store', 'wow', 'realli', 'good', 'could', 'solicit', 'opinion', 'peopl', 'use', '160', '180', 'day', 'day', 'worth', 'take', 'disk', 'size', 'money', 'hit', 'get', 'activ', 'display', 'realiz', 'real', 'subject', 'question', 'play', 'around', 'machin', 'comput', 'store', 'breifli', 'figur', 'opinion', 'somebodi', 'actual', 'use', 'machin', 'daili', 'might', 'prove', 'help', 'well', 'hellcat', 'perform', 'thank', 'bunch', 'advanc', 'info', 'could', 'email', 'post', 'summari', 'news', 'read', 'time', 'premium', 'final', 'around', 'corner', 'tom', 'willi', 'twilli', 'ecn', 'purdu', 'edu', 'purdu', 'electr', 'engin', 'convict', 'danger', 'enemi', 'truth', 'lie', 'nietzsch'])]



    data_token = data_json.map(lambda x: (x['label'],token_text(x['text'])))


    vocab = data_json.flatMap(lambda x: token_text(x['text'])).distinct().collect()
    print vocab[:10]

    ['fawn', 'k2b', '00011100b', 'darrylo', 'mbhi8bea', 'sonja', 'tilton', 'gag', '11546', 'phenomenologist']



    len(vocab)




    118149



We have a vocabulary of 118k+ words in our corpus, and we want to make a term frequency vector for each article.


    import numpy as np
    tf = data_token.map(lambda (label,tokens):(label,Counter(tokens))) \
                   .map(lambda (label,counter): (label,np.array([counter[word] if word in counter else 0 for word in vocab])))
        
    tf.cache()




    PythonRDD[192] at RDD at PythonRDD.scala:43



##TF-IDF
At this point I am going on a side question from what we were assigned.  We were told to fit a model using the Term Frequency, but I want to perform it with the normalized TF-IDF vectors.  These have historically done better for clustering concepts and ideas.   

The issue is that the most efficient way to do this in map-reduce/spark, requires that we have document ids.  We can not add them in a strait-forward way because we would have to put unique ides.  Because spark is sending each line to a different process, there isn't a reliable and scalable way to do this.   This is something we would need to do in the preprocessing.

Instead, we will do this a slower way through summing over columns.


    df = tf.map(lambda (label,tf): tf.astype(bool).astype(int))
    df.cache()




    PythonRDD[209] at RDD at PythonRDD.scala:43




    temp = df.reduce(lambda x,y:x+y)
    idf = np.log(13087./temp.astype(float))
    idf




    array([ 8.78622747,  8.78622747,  9.47937465, ...,  6.53493567,
            7.39993311,  9.47937465])




    idf.shape




    (118149,)




    tfidf = tf.map(lambda (label,tf): (label,tf*idf)).map(lambda (label,tfidf): (label,tfidf/np.linalg.norm(tfidf)))
    tfidf.cache()
    tfidf.first()




    (4, array([ 0.,  0.,  0., ...,  0.,  0.,  0.]))




    data = tfidf.map(lambda (label,tfidf):LabeledPoint(label,tfidf))
    trn,tst = data.randomSplit([0.7,.3])
    model = NaiveBayes.train(trn)


    results_rdd = trn.map(lambda x: (x.label,model.predict(x.features)))
    results_rdd.cache()
    results = results_rdd.map(lambda x: x[0]==x[1]).collect()
    float(sum(results))/len(results)




    0.9441328494446277



We were told to expect accuracies in the range of 80 - 87% using the Term Frequency method.  The TF-IDF method gave a much higher accuracy on the hold out set.   

Lets try to find the groups the model mis-classifies


    misclass1 = results_rdd.filter(lambda x: x[0] != x[1]).map(lambda x: (labels[x[0]],[labels[x[1]]]))
    misclass2 = results_rdd.filter(lambda x: x[0] != x[1]).map(lambda x: (labels[x[1]],[labels[x[0]]]))
    misclass = misclass1.union(misclass2)


    misclass_counts = misclass.reduceByKey(lambda x,y: x+y).map(lambda (group,arr): (group,Counter(arr)))


    for group, others in misclass_counts.collect():
        print "News Group:",group
        print "Total Mistakes:", sum(others.values())
        print "Misclassified as: ", others
        print ""

    News Group: sci.med
    Total Mistakes: 11
    Misclassified as:  Counter({u'soc.religion.christian': 3, u'talk.religion.misc': 3, u'sci.crypt': 1, u'comp.sys.mac.hardware': 1, u'misc.forsale': 1, u'talk.politics.guns': 1, u'sci.electronics': 1})
    
    News Group: comp.os.ms-windows.misc
    Total Mistakes: 40
    Misclassified as:  Counter({u'comp.sys.ibm.pc.hardware': 18, u'comp.graphics': 8, u'misc.forsale': 4, u'comp.sys.mac.hardware': 3, u'comp.windows.x': 2, u'sci.electronics': 2, u'sci.space': 2, u'rec.sport.hockey': 1})
    
    News Group: comp.windows.x
    Total Mistakes: 19
    Misclassified as:  Counter({u'comp.graphics': 8, u'comp.sys.ibm.pc.hardware': 2, u'comp.os.ms-windows.misc': 2, u'sci.space': 2, u'soc.religion.christian': 1, u'comp.sys.mac.hardware': 1, u'talk.religion.misc': 1, u'rec.sport.hockey': 1, u'rec.autos': 1})
    
    News Group: sci.crypt
    Total Mistakes: 31
    Misclassified as:  Counter({u'comp.graphics': 7, u'sci.electronics': 7, u'talk.politics.guns': 4, u'misc.forsale': 4, u'talk.politics.misc': 3, u'talk.religion.misc': 2, u'rec.motorcycles': 1, u'comp.sys.mac.hardware': 1, u'sci.med': 1, u'alt.atheism': 1})
    
    News Group: soc.religion.christian
    Total Mistakes: 191
    Misclassified as:  Counter({u'talk.religion.misc': 126, u'alt.atheism': 40, u'talk.politics.misc': 12, u'misc.forsale': 3, u'sci.med': 3, u'talk.politics.mideast': 3, u'rec.motorcycles': 1, u'comp.graphics': 1, u'comp.windows.x': 1, u'rec.sport.baseball': 1})
    
    News Group: comp.sys.ibm.pc.hardware
    Total Mistakes: 60
    Misclassified as:  Counter({u'comp.os.ms-windows.misc': 18, u'misc.forsale': 18, u'comp.graphics': 8, u'sci.electronics': 8, u'comp.sys.mac.hardware': 5, u'comp.windows.x': 2, u'talk.religion.misc': 1})
    
    News Group: rec.motorcycles
    Total Mistakes: 10
    Misclassified as:  Counter({u'rec.autos': 3, u'sci.electronics': 2, u'sci.crypt': 1, u'soc.religion.christian': 1, u'misc.forsale': 1, u'talk.religion.misc': 1, u'sci.space': 1})
    
    News Group: rec.autos
    Total Mistakes: 15
    Misclassified as:  Counter({u'misc.forsale': 7, u'rec.motorcycles': 3, u'sci.electronics': 2, u'comp.graphics': 1, u'comp.windows.x': 1, u'talk.politics.guns': 1})
    
    News Group: sci.space
    Total Mistakes: 21
    Misclassified as:  Counter({u'talk.religion.misc': 5, u'comp.graphics': 3, u'sci.electronics': 3, u'talk.politics.misc': 2, u'comp.windows.x': 2, u'comp.os.ms-windows.misc': 2, u'rec.motorcycles': 1, u'talk.politics.guns': 1, u'misc.forsale': 1, u'comp.sys.mac.hardware': 1})
    
    News Group: talk.politics.guns
    Total Mistakes: 75
    Misclassified as:  Counter({u'talk.politics.misc': 41, u'talk.religion.misc': 22, u'sci.crypt': 4, u'alt.atheism': 2, u'comp.sys.mac.hardware': 1, u'comp.graphics': 1, u'misc.forsale': 1, u'sci.med': 1, u'sci.space': 1, u'rec.autos': 1})
    
    News Group: comp.sys.mac.hardware
    Total Mistakes: 24
    Misclassified as:  Counter({u'comp.sys.ibm.pc.hardware': 5, u'sci.electronics': 4, u'misc.forsale': 4, u'comp.os.ms-windows.misc': 3, u'comp.graphics': 2, u'sci.med': 1, u'talk.politics.guns': 1, u'comp.windows.x': 1, u'sci.crypt': 1, u'sci.space': 1, u'rec.sport.baseball': 1})
    
    News Group: misc.forsale
    Total Mistakes: 52
    Misclassified as:  Counter({u'comp.sys.ibm.pc.hardware': 18, u'rec.autos': 7, u'comp.sys.mac.hardware': 4, u'sci.crypt': 4, u'rec.sport.hockey': 4, u'comp.os.ms-windows.misc': 4, u'soc.religion.christian': 3, u'sci.electronics': 3, u'talk.politics.guns': 1, u'comp.graphics': 1, u'sci.med': 1, u'sci.space': 1, u'rec.motorcycles': 1})
    
    News Group: rec.sport.baseball
    Total Mistakes: 8
    Misclassified as:  Counter({u'rec.sport.hockey': 4, u'soc.religion.christian': 1, u'comp.sys.mac.hardware': 1, u'talk.politics.misc': 1, u'talk.politics.mideast': 1})
    
    News Group: talk.politics.misc
    Total Mistakes: 67
    Misclassified as:  Counter({u'talk.politics.guns': 41, u'soc.religion.christian': 12, u'talk.politics.mideast': 6, u'sci.crypt': 3, u'sci.space': 2, u'talk.religion.misc': 1, u'rec.sport.hockey': 1, u'rec.sport.baseball': 1})
    
    News Group: comp.graphics
    Total Mistakes: 42
    Misclassified as:  Counter({u'comp.windows.x': 8, u'comp.sys.ibm.pc.hardware': 8, u'comp.os.ms-windows.misc': 8, u'sci.crypt': 7, u'sci.space': 3, u'comp.sys.mac.hardware': 2, u'sci.electronics': 2, u'soc.religion.christian': 1, u'misc.forsale': 1, u'talk.politics.guns': 1, u'rec.autos': 1})
    
    News Group: talk.religion.misc
    Total Mistakes: 201
    Misclassified as:  Counter({u'soc.religion.christian': 126, u'alt.atheism': 37, u'talk.politics.guns': 22, u'sci.space': 5, u'sci.med': 3, u'sci.crypt': 2, u'talk.politics.mideast': 2, u'talk.politics.misc': 1, u'rec.motorcycles': 1, u'comp.windows.x': 1, u'comp.sys.ibm.pc.hardware': 1})
    
    News Group: talk.politics.mideast
    Total Mistakes: 17
    Misclassified as:  Counter({u'talk.politics.misc': 6, u'alt.atheism': 5, u'soc.religion.christian': 3, u'talk.religion.misc': 2, u'rec.sport.baseball': 1})
    
    News Group: rec.sport.hockey
    Total Mistakes: 12
    Misclassified as:  Counter({u'misc.forsale': 4, u'rec.sport.baseball': 4, u'comp.os.ms-windows.misc': 1, u'talk.politics.misc': 1, u'comp.windows.x': 1, u'sci.electronics': 1})
    
    News Group: alt.atheism
    Total Mistakes: 85
    Misclassified as:  Counter({u'soc.religion.christian': 40, u'talk.religion.misc': 37, u'talk.politics.mideast': 5, u'talk.politics.guns': 2, u'sci.crypt': 1})
    
    News Group: sci.electronics
    Total Mistakes: 35
    Misclassified as:  Counter({u'comp.sys.ibm.pc.hardware': 8, u'sci.crypt': 7, u'comp.sys.mac.hardware': 4, u'sci.space': 3, u'misc.forsale': 3, u'rec.motorcycles': 2, u'comp.graphics': 2, u'comp.os.ms-windows.misc': 2, u'rec.autos': 2, u'sci.med': 1, u'rec.sport.hockey': 1})
    


###Conclusion

The majority of the misclassification comes from the regiously themeed groups, or the similar tech groups.   These are not surprising results.   To be honest, I am constently impressed with how successful these NLP methods are at categorizing and themeing information in a way that is both predictive and explanitory. 

##Word2Vec

This is one of my favorite tools for fun, but not profit (yet!!!).   I will show you what I mean.  


    data = sc.textFile('s3n://AKIAJRH3ZAWFYUFN5WIA:xSztE4PvK7GQ3zHuohSoOgrOdV9OosfP+4WXod0R@sparkdatasets/text8_lines')


    data.count()




    1062826




    data_token = data.map(lambda x: token_text(x))
    data_token.cache()
    data_token.first()




    ['anarch',
     'origin',
     'term',
     'abus',
     'first',
     'use',
     'earli',
     'work',
     'class',
     'radic',
     'includ']




    from pyspark.mllib.feature import Word2Vec
    word2vec = Word2Vec()
    model = word2vec.fit(data_token)


    def add_words(word1,word2):
        vec1 = model.transform(token_text(word1.lower())[0])
        vec2 = model.transform(token_text(word2.lower())[0])
        vec1p2 = vec1 + vec2
        print "Word 1 + Word 2"
        print model.findSynonyms(vec1p2,1)
        vec1m2 = vec1 - vec2
        print "Word 1 - Word 2"
        print model.findSynonyms(vec1m2,1)
        vec2m1 = vec2 - vec1
        print "Word 2 - Word 1"
        print model.findSynonyms(vec2m1,1)


    add_words('politics','anarchism')

    Word 1 + Word 2
    [(u'libertarian', 3.4323648130450475)]
    Word 1 - Word 2
    [(u'reichskammergericht', 1.0360935689992155)]
    Word 2 - Word 1
    [(u'individualist', 1.5454184699053521)]


Lets unpack what this has done.

Word2Vec fits a neural network on words next to each other, and produces a feature vecture based on the fit.  We then transform two words into vectors. 

The first vector is "Politics" + "Anarchism" and we get back a vector "Libertarian".   That's a pretty good fit.

The second vector is "Politics" - "Anarchism" and we get back a vector "Reichskammergericht".   A google search tells us what this is:

> The Reichskammergericht (English: Imperial Chamber Court Latin: Iudicium imperii) was one of two highest judicial institutions in the Holy Roman Empire

The final vetors is "Anarchism" - "Politics" and we get back a vector "Individualist".

Given a corpus, word2vec does a good job of capturing meaning of words relative to other words.   How fun is this!!!!

