Title: Galvanize - Week 01 - Day 4
Date: 2015-06-04 10:20
Modified: 2015-06-04 10:30
Category: Galvanize
Tags: data-science, galvanize, sql, postgresql, psycopg2
Slug: galvanize-data-science-01-04
Authors: Bryan Smith
Summary: The fourth day of Galvanize's Immersive Data Science program in San Francisco, CA where we got an in depth introduction to Advance SQL and PostgreSQL on Facebook style tables.


#Galvanize Immersive Data Science

##Week 1 - Day 4

The day started out with a mini-quiz on object-oriented programming, and that was followed by an introduction to git and sophisticated join queries.   Our instructor for the day used to work at Facebook, and she walked us through some the queries she would do on the job.  

She then gave us a simulated data set that match the structure, but not the content, of Facebook tables and we had an individual sprint attempting to complete 10 queries in 2 hours.

After lunch we had a lecture on pyscopg2, a python library to use to connect and interact with a PostgreSQL server.   We ran a server locally, loaded with the same data as the morning, and were given an assignment to construct a pipeline that we could run each day to give us an updated status of our users.   We were to check on results for today being set to Aug 14, 2014. 

Our resulting script is below:


    import psycopg2
    from datetime import datetime
    
    conn = psycopg2.connect(dbname='socialmedia', user='postgres', password='password', host='localhost')
    c = conn.cursor()
    
    today = '2014-08-14'
    
    timestamp = datetime.strptime(today, '%Y-%M-%d').strftime("%s")
    
    c.execute(
        '''CREATE TABLE logins_7d_%s AS
        WITH
        main AS (
        SELECT
            r.userid,
            tmstmp::date AS reg_date,
            CASE WHEN optout.userid IS NULL then 0 ELSE 1 END AS opt_out
        FROM registrations r
        LEFT OUTER JOIN optout
        ON r.userid = optout.userid
        ORDER BY r.userid),
        last AS (
        SELECT
            userid,
            MAX(tmstmp::date) AS last_login
        FROM logins
        GROUP BY userid
        ORDER BY userid),
        last7 AS (
        SELECT
            t.userid,
            COUNT(t.dt) AS logins_7d
        FROM (
            SELECT
                DISTINCT userid,
                tmstmp::date AS dt
            FROM logins
            WHERE logins.tmstmp > timestamp '2014-08-14' - interval '7 days'
            GROUP BY userid, tmstmp::date
            ORDER BY userid) t
        GROUP BY t.userid),
        last7m AS (
        SELECT t.userid, COUNT(t.dt) AS logins_7m
        FROM (
            SELECT
                DISTINCT userid,
                tmstmp::date AS dt
            FROM logins
            WHERE
                logins.tmstmp > timestamp '2014-08-14' - interval '7 days' AND
                logins.type = 'mobile'
            GROUP BY userid, tmstmp::date
            ORDER BY userid) t
        GROUP BY t.userid),
        last7w AS (
        SELECT
            t.userid,
            COUNT(t.dt) AS logins_7w
        FROM (
            SELECT
                DISTINCT userid,
                tmstmp::date AS dt
            FROM logins
            WHERE
                logins.tmstmp > timestamp '2014-08-14' - interval '7 days' AND
                logins.type = 'web'
            GROUP BY userid, tmstmp::date
            ORDER BY userid) t
        GROUP BY t.userid),
        uf1 AS (
        (SELECT * FROM friends)
        UNION ALL
        (SELECT userid2, userid1 FROM friends)),
        uf2 AS (
        SELECT DISTINCT *
        FROM uf1),
        friend_cnt AS (
        SELECT
            userid1 AS userid,
            COUNT(1) AS num_friends
        FROM uf2
        GROUP BY userid)
        SELECT
            main.userid,
            reg_date,
            last_login,
            coalesce(logins_7d,0) AS logins_7d,
            coalesce(logins_7m,0) AS logins_7d_mobile,
            coalesce(logins_7w,0) AS logins_7d_web,
            coalesce(num_friends,0) AS num_friends,
            opt_out
        FROM main
        LEFT OUTER JOIN last
        ON main.userid = last.userid
        LEFT OUTER JOIN last7
        ON main.userid = last7.userid
        LEFT OUTER JOIN last7m
        ON main.userid = last7m.userid
        LEFT OUTER JOIN last7w
        ON main.userid = last7w.userid
        LEFT OUTER JOIN friend_cnt
        ON main.userid = friend_cnt.userid;''' % timestamp
    )
    
    conn.commit()
    conn.close()

We also learned how pull data and load the data into a pandas dataframe.   


    from pandas.io import sql
    from pandas.io.sql import read_sql
    
    conn = psycopg2.connect(dbname='socialmedia', 
                            user='postgres', 
                            password='password', 
                            host='localhost')
    
    sql = 'SELECT * FROM logins_7d_1389686880 LIMIT 20;'
    
    df = read_sql(sql, conn, index_col="userid", coerce_float=True, params=None)
    
    conn.close()
    
    df




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reg_date</th>
      <th>last_login</th>
      <th>logins_7d</th>
      <th>logins_7d_mobile</th>
      <th>logins_7d_web</th>
      <th>num_friends</th>
      <th>opt_out</th>
    </tr>
    <tr>
      <th>userid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-06-23</td>
      <td>2014-08-13</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-12-21</td>
      <td>2014-08-12</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-04-18</td>
      <td>2014-08-14</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-17</td>
      <td>2014-08-13</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-11</td>
      <td>2014-08-09</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2013-08-31</td>
      <td>2014-08-10</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2013-08-18</td>
      <td>2014-08-12</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2014-03-21</td>
      <td>2014-08-12</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2014-05-03</td>
      <td>2014-08-11</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014-06-06</td>
      <td>2014-08-11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2013-08-31</td>
      <td>2014-08-10</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2013-08-16</td>
      <td>2014-08-10</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2013-09-12</td>
      <td>2014-08-13</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014-07-29</td>
      <td>2014-08-14</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2013-11-03</td>
      <td>2014-08-11</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2013-10-09</td>
      <td>2014-08-13</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2014-02-16</td>
      <td>2014-08-12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2014-04-20</td>
      <td>2014-08-14</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2014-07-02</td>
      <td>2014-08-12</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2014-08-14</td>
      <td>2014-05-10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Today was a very intense day.   But the programming is delivering on what it promised: Hands On Learning From Experience Professions!


    
