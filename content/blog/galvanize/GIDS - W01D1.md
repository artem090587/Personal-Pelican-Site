Title: Galvanize - Week 01 - Day 1
Date: 2015-06-01 10:20
Modified: 2015-06-01 10:30
Category: Galvanize
Tags: data-science, galvanize, python
Slug: galvanize-data-science-01-01
Authors: Bryan Smith
Summary: My experience attending the first day of Galvanize's Immersive Data Science program in San Francisco, CA

#Galvanize Immersive Data Science

##Week 1 - Day 1

Today is my first day attending Galvanize's Immersive Data Science Program in San Francisco, CA.   The program is a 12 week program that is approximately 10 hours a day of learning and activities to reinforce and refine the learning.   I am very excited to be a part of this program.

##My Background

I have a Ph.d in Theoretical High Energy Particle Physics and Cosmology, earned a Data Analysis Nano-degree from Udacity.com, and also am current working on a M.S. in Computer Science from GA Tech.    I have also spent the last 8 years teaching high school physics and robotics.   

I will likely have some strength with math and theory, but I have no doubt that my programming will significantly improve over the next 12 weeks.   Everyone in the program is well educated and intelligent, and each one of them have strengths in some areas and room for improvements in others.  It seems to be a strength for this program.   No matter your weakness, there are students that have that as a strength.


## Summary of the Day

The first half of the day is getting to know our instructors, hour cohort, and the amazingly nice galvanize complex.  It is a 5 story building filled with startup companies, work spaces, galvanize students, and other tech visitors.   I found this to be an impressive building.

After about an hour of getting to know everyone, we were given a presentation.  That was followed by a tour of the galvanize building.  We were then given an assessment on the pre-course material that was given to us before we showed up.

After lunch, we had an 90 minute lecture, then worked on a paired sprint assignment for about 3 hours.   This assignment involved...

After we finished the sprint, we were invited to a Galvanize happy hour to socialize over beer and wine.   I feel very lucky to be apart of this program, and have been impressed with my cohort, the instructors, and Galvanize.  

## The Test

The pre-course material required us to complete material on the following topics:

1.  Python
2.  Linear Algebra
3.  SQL
4.  Numpy/Pandas
5.  Probability
6.  Statistics
7.  Hypothesis Testing
8.  Web Awareness

The initial assessment we were given was a 120 minute test on the first 5 topics.  It was an 'open book' test, but that does not mean it was easy.  A majority of my peers did not finish within the allotted time.   

I am not going to post details on the test because I would hate to ruin the thrill of discovery for potential future students.

## LUNCH!!!!

They provide us a lunch on the first day, but most days we have an 75 minute break for lunch.  There are kitchen, fridges, storage for us to use if we wish.   The lunch was nice, from a local Thai place.   

## Lecture

The lecture, in my opinion, was a little redundant with the course material.  It seemed structure under the assumption that you didn't read or review the python pre-course materials.  I understand its important that everyone is on the same starting point, but I wish we got to jump in a little deeper.

I did learn and see the importance of using generators when possible.   It save both memory and time.   

## Paired programming

After the lecture we grouped up for a paired programming assignment.   We trade off roles of being the driver and the navigator in 20 to 30 minute rotations for a 3 hour block of programming.  We start of by forking the day's assignment from a Github repo and cloning it locally.  Today we then worked two projects.  The first project was completing a list of functions based on a description of the function, including inputs and outputs.  The second project was fixing inefficiently running code.

A simple example is checking if a key is in a dictionary.   Before today I might have checked to see if it was in the keys() results, but we can see that for medium size dictionaries that it is almost 40x slower than just using in in the dictionary.


    from collections import Counter
    from random import randint
    
    cnt = Counter([randint(0,1000) for x in range(10000)])
    %timeit 1 in cnt.keys()
    %timeit 1 in cnt

    100000 loops, best of 3: 4.17 µs per loop
    10000000 loops, best of 3: 111 ns per loop


We saw similar results for iter compared to iteritems, range to xrange, and izip to zip.   It was a useful assignment for the content and the practice of collaborating with someone else.


## After reception

Galvanize SF now runs two cohorts 6 weeks apart.  We had a mixer with previous cohort, enjoying beer, wine, and conversation on the roof of the building.  

After 11 hours at Galvanize, I decided it was time to head home.   Definitely looking forward to day 2. 




    
