Title: Galvanize - Week 07 - Day 2
Date: 2015-07-14 10:20
Modified: 2015-07-14 10:30
Category: Galvanize
Tags: data-science, galvanize, aws, parallel programing, s3, ec2
Slug: galvanize-data-science-07-02
Authors: Bryan Smith
Summary: Today we covered high performance programing and aws

#Galvanize Immersive Data Science

##Week 7 - Day 2

Today we started our process of large scale and cloud based data science.  Our 'quiz' was to set up an AWS account and link it to Galvanize.  

Our morning lecture was on AWS, boto package for python, and details of S3, EBS, and EC2.

##S3 on AWS

Our first task was to use the web interface and set up an S3 bucket, and I named my 'galvanizeweek7'.   To make the data web accessible, we needed to change the permissions of either the file or the bucket.  

We were given this permission to use on our bucket:

        {
          "Version": "2008-10-17",
          "Statement": [
            {
               "Sid": "AllowPublicRead",
               "Effect": "Allow",
               "Principal": {
                  "AWS": "*"
               },
               "Action": "s3:GetObject",
               "Resource": "arn:aws:s3:::galvanizeweek7/*"
               }
          ]
         }

We then uploaded a cancer data set into our bucket.


    import pandas as pd
    cancer = pd.read_csv('https://s3.amazonaws.com/galvanizeweek7/cancer.csv')
    cancer.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cancer</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>559</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>681</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>746</td>
    </tr>
  </tbody>
</table>
</div>



Now that we can load data from our S3 bucket, we are going to use boto upload an analysis to the bucket we downloaded the data.


    cancer['rate'] = cancer.cancer/cancer.population
    cancer.to_csv("cancer_with_rate.csv", index=False)
    import matplotlib.pyplot as plt
    
    plt.hist(cancer.rate, bins=30)
    plt.xlabel("Cancer Rate")
    plt.ylabel("Count")
    plt.savefig('cancer_rate.jpg')


    import boto
    conn = boto.connect_s3(aws_access_key_id,aws_secret_access_key)
    bucket = conn.get_bucket('galvanizeweek7')
    csvfile = bucket.new_key('cancer_with_rate.csv')
    csvfile.set_contents_from_filename('cancer_with_rate.csv')
    imgfile = bucket.new_key('cancer_rate_hist.jpg')
    imgfile.set_contents_from_filename('cancer_rate.jpg')




    54458




    from IPython.display import Image
    Image(url='https://s3.amazonaws.com/galvanizeweek7/cancer_rate_hist.jpg')




<img src="https://s3.amazonaws.com/galvanizeweek7/cancer_rate_hist.jpg"/>



That is pretty cool in my book.  Our next challenge is to run this analysis on an AWS EC2 instance.

##EC2 on AWS

First we need to launch an EC2 instance.  There is a community data science image that we were instructed to use.  Just for the record it is:

$$\mbox{AMI: ami-d1737bb8}$$

After we lauched the image we downloaded our key-pair pem file locally so that we can ssh into the instance.  We also need to upload our file for analysis using scp.

$$\mbox{ssh -X -i keypair.pem User@Domain}$$

$$\mbox{scp -i keypair.pem /path/to/myfile.txt User@Domain:}$$

Notice the ":" in the last line.  I waisted 5 minutes by not including it. 

Using these steps I uploaded the followling contents to 'analysis.py' onto my EC2 instance and ran 'python analysis.py'.   


    def do_analysis():
        import boto
        aws_access_key_id = "KEY"
        aws_secret_access_key = "SECRET"
        conn = boto.connect_s3(aws_access_key_id,aws_secret_access_key)
        galvanzieweek7copy = conn.create_bucket('galvanzieweek7copy')
        
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
    
        cancer = pd.read_csv('https://s3.amazonaws.com/galvanizeweek7/cancer.csv')
        cancer['rate'] = cancer.cancer/cancer.population
        cancer.to_csv("cancer_with_rate.csv", index=False)
        plt.hist(cancer.rate)
        plt.xlabel("Cancer Rate")
        plt.ylabel("Count")
        plt.savefig('cancer_rate.jpg')
        
        csvfile = galvanzieweek7copy.new_key('cancer_with_rate.csv')
        csvfile.set_contents_from_filename('cancer_with_rate.csv')
        imgfile = galvanzieweek7copy.new_key('cancer_rate_hist.jpg')
        imgfile.set_contents_from_filename('cancer_rate.jpg')
    
        import os
        os.remove('cancer_with_rate.csv')
        os.remove('cancer_rate.jpg')
        
    if __name__ == "__main__":
        do_analysis()


    Image(url='https://s3.amazonaws.com/galvanzieweek7copy/cancer_rate_hist.jpg')




<img src="https://s3.amazonaws.com/galvanzieweek7copy/cancer_rate_hist.jpg"/>



We see two things:

1.  I forgot to set the number of bins
2.  I misspelled galvanize.   

Other then those two mistakes, I ran an entire 'job' on the cloud.   I'm sure there is a joke about getting high..... I just can't think of it.


##Parallelism 

Our afternoon lectures were on multicore and threaded processes in python.   We did not deal with communcation through queues between seperate threads.  That would have been more interesting for me, but most of my cohort has not had graduate level System architecture classes.   C'est la vie.  I still learned a lot, and sometimes its about getting neurons to fire after being dormant for a while.  

Lets jump in...

We are going perfor a computational intensive task, checking if a number is prime by trying to divide it by every number between 2 and its square root.   We will do this for a range of numbers, getting a list of primes.   

Since each number's check is independant of another number's check, we can break this up and run them separately.  We are timing two processes:

1.  Finding every prime between 100000000 and 101000000 in squence
2.  Breaking the job into for pieces, and running each job separately.   
    1.  I only have two cores on this machine, so the 4 is really a 2 as far as performance increase.  There are 4 jobs, but they are not getting 4x improvmment (minus overhead of setup)


    import math
    import multiprocessing
    import itertools
    from timeit import Timer
    
    
    def check_prime(n):
        if n % 2 == 0:
            return False
        for i in xrange(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    
    def primes_sequential():
        primes = []
        number_range = xrange(100000000, 101000000) 
        for possible_prime in number_range:
            if check_prime(possible_prime):
                primes.append(possible_prime)
        print "Number of Primes Found: ", len(primes) # primes[:10], primes[-10:]
    
    
    def primes_in_range(range):
        primes = []
        for possible_prime in range:
            if check_prime(possible_prime):
                primes.append(possible_prime)
        return primes
        
    def primes_parallel():
        num_processes = 4
        start = 100000000
        end = 101000000
        diff = end-start
        range_list = [xrange(100000000+i*diff/num_processes,start+(i+1)*diff/num_processes) for i in range(num_processes)]
        
        pool = multiprocessing.Pool(processes=num_processes)
        outputs = pool.map(primes_in_range, range_list)
        primes = []
        for result in outputs:
            primes.extend(result)
        print "Number of Primes Found: ", len(primes) #, primes[:10], primes[-10:]
    
    #if __name__ == "__main__":
    t = Timer(lambda: primes_sequential())
    print "Completed sequential in %s seconds." % t.timeit(1)
    
    t = Timer(lambda: primes_parallel())
    print "Completed parallel in %s seconds." % t.timeit(1)

    Number of Primes Found:  54208
    Completed sequential in 22.6053831577 seconds.
    Number of Primes Found:  54208
    Completed parallel in 10.060131073 seconds.


Running thee following line in command line after running this code shows that there are 4 jobs.  When I set it for 20, there were 20 jobs.   The performance increase, however, was not there because we are limited by my computer power.

>ps aux | grep primes.py

##Concurrency

Concurrency is about different threads on the same core, but switching between those threads when there is downtime.   The previous problem is not a good example because the computation does not have much down time.   A request for a website, however, has downtime between making the response and recieving the request.   That is prime time to be doing other computations.

For this part of our afternoon sprint we are using [`threading`](https://docs.python.org/2/library/threading.html).

We are going to make request of hacker news in a sequence and threaded way.   My intution is to see a great improvement. 


    import multiprocessing
    import requests
    import sys
    import threading
    from timeit import Timer
    
    
    def request_item(item_id):
        try:
            r = requests.get("http://hn.algolia.com/api/v1/items/%s" % item_id)
            print "Thread: ", threading.currentThread().getName()
            return r.json()
        except requests.RequestException:
            return None
    
    
    def request_sequential():
        sys.stdout.write("Requesting sequentially...\n")
        results = []
        for item in range(1,21):
            results.append(request_item(item))
        sys.stdout.write("done.\n")
        return results
    
    def request_concurrent():
        sys.stdout.write("Requesting in parallel...\n")
        threads = [threading.Thread(target=target_function, args=(i,)) for i in range(1,21)]
        for t in threads: t.start()
        for t in threads: t.join()
        sys.stdout.write("done.\n")
        return [t.result for t in threads]
    
    def target_function(item):
        self = threading.current_thread()
        self.result = request_item(item)
    
        
    
    #if __name__ == '__main__':
    t = Timer(lambda: request_sequential())
    print "Completed sequential in %s seconds." % t.timeit(1)
    print "--------------------------------------"
    
    t = Timer(lambda: request_concurrent())
    print "Completed using threads in %s seconds." % t.timeit(1)

    Requesting sequentially...
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    Thread:  MainThread
    done.
    Completed sequential in 7.00593900681 seconds.
    --------------------------------------
    Requesting in parallel...
    Thread:  Thread-10
    Thread:  Thread-12
    Thread:  Thread-11
    Thread:  Thread-13
    Thread:  Thread-14
    Thread:  Thread-9
    Thread:  Thread-25
    Thread:  Thread-18
    Thread:  Thread-24
    Thread:  Thread-16
    Thread:  Thread-27
    Thread:  Thread: Thread-15 
    Thread-17
    Thread:  Thread-23
    Thread:  Thread-22
    Thread:  Thread-26
    Thread:  Thread-28
    Thread:  Thread-21
    Thread:  Thread-19
    Thread:  Thread-20
    done.
    Completed using threads in 0.275010824203 seconds.


In the sequential model, we have a first in first out process.  We get the requests in the order they are programmed.

In the threaded model, we see that they run in a 'random' order.  We see that thread 15 and thread 17 ran at nearly the same time by how it is displayed in the print stream.  We also see that we go from 7 seconds to almost a quarter of a second to make and receive all the processes.   This is a much more efficent of my computer's resources.  The gain, again, is due to the fact that there is a long wait time in making internet requests.

5. Run `request_sequential` and `request_concurrent` and compare the run time.

##Parallelism and Concurrency

In web scraping we may have different categories we want to scrape, but those scrapings can be threaded.  This is an opportunity to combine parallelism and concurrency. 

We are going to scrape the Yelp API for 4 cities, one on each core, and thread the pull of 20 businesses from each city.  Instead of print or returning the data, we are going to write it dirrectly to a local MongoDB database.  


    KEY = "OoJlpKgaVfdWK97h4RI4fA"
    SECRET_KEY = "ZuOdCWb-rUAjOcHpw9r6Hz6wDKc"
    TOKEN = "oARQTS19fvi6R4uFWvNsVrjCB0cjduqR"
    SECRET_TOKEN = "2IZtIHSG1RgCuNDmO17zicYC3ac"


    import json
    import oauth2
    import urllib
    import urllib2
    
    
    def request(host, path, url_params=None):
        """Prepares OAuth authentication and sends the request to the API.
        Args:
            host (str): The domain host of the API.
            path (str): The path of the API after the domain.
            url_params (dict): An optional set of query parameters in the request.
        Returns:
            dict: The JSON response from the request.
        Raises:
            urllib2.HTTPError: An error occurs from the HTTP request.
        """
        url_params = url_params or {}
        url = 'http://{0}{1}?'.format(host, urllib.quote(path.encode('utf8')))
    
        consumer = oauth2.Consumer(KEY, SECRET_KEY)
        oauth_request = oauth2.Request(method="GET", url=url, parameters=url_params)
    
        oauth_request.update(
            {
                'oauth_nonce': oauth2.generate_nonce(),
                'oauth_timestamp': oauth2.generate_timestamp(),
                'oauth_token': TOKEN,
                'oauth_consumer_key': KEY
            }
        )
        token = oauth2.Token(TOKEN, SECRET_TOKEN)
        oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, token)
        signed_url = oauth_request.to_url()
    
        #print u'Querying {0} ...'.format(signed_url)
    
        conn = urllib2.urlopen(signed_url, None)
        try:
            response = json.loads(conn.read())
        finally:
            conn.close()
    
        return response


    from timeit import Timer
    from pymongo import MongoClient
    import multiprocessing
    import threading
    
    POOL_SIZE = 4
    API_HOST = "api.yelp.com"
    SEARCH_PATH = '/v2/search'
    BUSINESS_PATH = '/v2/business/'
    
    DB_NAME = "yelp"
    COLLECTION_NAME = "business"
    
    client = MongoClient()
    db = client.yelp
    coll = db.scrape
    
    
    def city_search_parallel(city):
        """
        Retrieves the JSON response that contains the top 20 business meta data for city.
        :param city: city name
        """
        response = city_search(city)
        business_ids = [x['id'] for x in response['businesses']]
        business_info_concurrent(business_ids)
    
    
    
    
    def business_info_concurrent(ids):
        """
        Extracts the business ids from the JSON object and
        retrieves the business data for each id concurrently.
        :param json_response: JSON response from the search API.
        """
        threads = [threading.Thread(target=business_info, args=(id,)) for id in ids]
        for t in threads: t.start()
        for t in threads: t.join()
    
    
    def scrape_parallel_concurrent(pool_size):
        """
        Uses multiple processes to make requests to the search API.
        :param pool_size: number of worker processes
        """
        coll.remove({})  # Remove previous entries from collection in Mongodb.
        with open('data/cities') as f:
            cities = f.read().splitlines()
            pool = multiprocessing.Pool(processes=pool_size)
            outputs = pool.map(city_search_parallel, cities)
    
    
    def business_info(business_id):
        """
        Makes a request to Yelp's business API and retrieves the business data in JSON format.
        Dumps the JSON response into mongodb.
        :param business_id:
        """
        business_path = BUSINESS_PATH + business_id
        response = request(API_HOST, business_path)
        coll.insert(response)
    
    
    def city_search(city):
        """
        Makes a request to Yelp's search API given the city name.
        :param city:
        :return: JSON meta data for top 20 businesses.
        """
        params = {'location': city, 'limit': 20}
        json_response = request(API_HOST, SEARCH_PATH, url_params=params)
        return json_response
    
    
    def scrape_sequential():
        """
        Scrapes the business's meta data for a list of cities
        and for each business scrapes the content.
        """
        coll.remove({})  # Remove previous entries from collection in Mongodb.
        with open('data/cities') as f:
            cities = f.read().splitlines()
            for city in cities:
                response = city_search(city)
                business_ids = [x['id'] for x in response['businesses']]
                for business_id in business_ids:
                    business_info(business_id)
    
    
    if __name__ == '__main__':
        t = Timer(lambda: scrape_sequential())
        print "Completed sequential in %s seconds." % t.timeit(1)
    
        t2 = Timer(lambda: scrape_parallel_concurrent(POOL_SIZE))
        print "Completed parallel in %s seconds." % t2.timeit(1)

    Completed sequential in 34.240404129 seconds.
    Completed parallel in 3.13069486618 seconds.


Instead of taking 30 seconds to scrape 80 businesses from 4 cities sequentially, we can get the same information in our database by running on multiple cores, and each core threading highly latent requests, in 3 seconds.  

And fianlly, lets just make sure we have all 80 businesses in our database:


    coll.find().count()




    80


