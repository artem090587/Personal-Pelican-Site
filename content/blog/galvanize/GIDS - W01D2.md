Title: Galvanize - Week 01 - Day 2
Date: 2015-06-02 10:20
Modified: 2015-06-02 10:30
Category: Galvanize
Tags: data-science, galvanize, python
Slug: galvanize-data-science-01-02
Authors: Bryan Smith
Summary: My second day of Galvanize's Immersive Data Science program in San Francisco, CA where I learned about Object Oriented Programming, scoping in python, and built a black jack game. 

#Galvanize Immersive Data Science
##Week 1 - Day 2

Today was he first 'regular' day in the program.  I showed up about 90 minutes before the mini-quiz to review the readings for the day's lecture on Object Oriented Programming (OOP).   At 9:30 we started the mini-quiz on SQL statements and results.  I found it rather simple.  We were given 30 minutes to complete it, and I finished in about 10 minutes.  Most the topics involved analogs in pandas that I am familiar with, so I think that's why I finished rather quickly.

##Lecture
We had two lectures today.  The first lecture was on object oriented structures, and how to implement them in python.  The afternoon lecture was on scoping in python, and a little bit of debugging.  We were introduced to pdb, but told that the use of debuggers is not well integrated in the data science community.

### LEGB
We were told the variables are looked for in the order of local, enclosing function, global, and python build-in.   I made a set of functions to try to illustrate it for myself.



    x = 5
    def printer():
        print x  #globally finds x
    printer()
    
    def printer1():
        x = -1
        print x  #locally finds x
    printer1()
    
    def printer2():
        def innerprinter():
            print x  #globally finds x
        innerprinter()
    printer2()
    
    def printer3():
        x=3
        def innerprinter():
            print x #encapsulating function finds x
        innerprinter()
    printer3()
    
    def printer4():
        def innerprinter():
            print int #built-in function finds int
        innerprinter()
    printer4()
    
    def printer5():
        int = 3
        def innerprinter():
            print int #encapsulating function finds int
        innerprinter()
    printer5()
    
    int = 6
    def printer6():
        def innerprinter():
            print int #globally finds int
        innerprinter()
    printer6()

    5
    -1
    5
    3
    <type 'int'>
    3
    6


## Paired Programming Sprint

Today's project involved programming a text based game of black jack with a dealer and 1 number of players.  We also had the extra credit options adding n-players, AI/Bot players, double down, and split.  I am happy to report that we that my partner and I were able to complete the first three, but ran out of time before implementing split.

We started off with pencil and paper using the noun,verb method of abstraction.   We settled on making a Deck, a Player, and Hand, and Game, and an AI.    Its clear at the end that we should have abstracted the game more, and given the Hand class more responsibilities to best implement the split method.

After we finished we had our dealer hit until 17 or above, while our AI bot hit until soft 17 below.   We also had it implement a doubling bettering strategy.   In our sample, the AI agent one more often then not came out ahead from this setup.   It added a little credence to the ways dealer's seem to play in Las Vegas.

The repo is currently private, because it could be a project for future cohorts.   I do not want to make a copy public, but will if they give permission.




    
