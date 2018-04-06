# topic_model_LDA

**Context**

This contains data of news headlines published over a period of 15 years. From the reputable Australian news source ABC (Australian Broadcasting Corp.)

Site: http://www.abc.net.au/

Prepared by Rohit Kulkarni

**Content**

Format: CSV Rows: 1,103,665
Column 1: publish_date (yyyyMMdd format)
Column 2: headline_text (ascii, lowercase)
Start Date: 2003-02-19 End Date: 2017-12-31

**Description**

This includes the entire corpus of articles published by the ABC website in the given time range. With a volume of 200 articles per day and a good focus on international news, we can be fairly certain that every event of significance has been captured here.

Digging into the keywords, one can see all the important episodes shaping the last decade and how they evolved over time. Ex: financial crisis, iraq war, multiple US elections, ecological disasters, terrorism, famous people, Australian crimes etc.

**Tasks Performed**

1. Data Profiling and Exploration for Most Used Words
2. Stop Words Elimination
3. Basic LDA (Latent Dirichlet Allocation) with 10,000 data points
4. Larger LDA with 100,000 data points
5. Year-by-year analysis of relevant topics
