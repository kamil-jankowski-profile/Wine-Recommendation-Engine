

"""
Project: Wine Recommendation Engine
Author: Kamil Jankowski
Contact: https://www.linkedin.com/in/kamil-jankowski-bb8b38165
Created: April 2020

Description:

Recommendation system helps buyers to find relevant wine bottles to save their time and money.
Python's algorithms build co-occurrence matrix and based on description, country, province of origin, variety, reviews (0-100 points) and price segment of wine return from database the most similar products.  
    
"""

# Loading basic packages

import pandas as pd
import numpy as np
import matplotlib as plt
pd.set_option("display.max.columns", None)

# Insert the path to folder with winemag-data-130k-v2.csv on your desktop
metadata = pd.read_csv('C:/Users/winemag-data-130k-v2.csv', low_memory=False)
metadata.head()

# We have "nan" string values, let's remove them and copy columns which will be needed to build engines

wine = metadata[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery']]
wine = wine.query("title != 'NaN' and country != 'NaN' and description != 'NaN' and points != 'NaN' and price != 'NaN' and province != 'NaN' and variety != 'NaN' and winery != 'NaN'")
wine = wine.dropna()

# sort values by points
wine = wine.sort_values('points', ascending=False)
wine.head()
wine.shape

# We have 120 915 observations in 8 columns which is too much to calculate cosine similarity on mine (and most of us) computer so let's use only part of the original database


wine['points'].describe()
quantile = wine["points"].quantile(0.80)
wine = wine.copy().loc[wine["points"] >= quantile]
wine.shape
# Ok, we have 25% of the original database. It should be enough to prepare reliable recommendation engine. If you have more powerful computer, feel free to change used quantile


# Let's build the first engine which will be based on description of the wines bottles
wine["description"].head()


from sklearn.feature_extraction.text import TfidfVectorizer

# To increase accuracy, we remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(wine['description'])


from sklearn.metrics.pairwise import linear_kernel
# Below code will create the cosine similarity matrix which be needed to build engine.
# Be patient, calculation of it would takes a few seconds. If you get memory error, increase quantile value used to divide dataframe.

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(wine.index, index=wine['title']).drop_duplicates()


def wine_recommendations(title, cosine_sim=cosine_sim):
    
    # fit index of the wine to the title
    idx = indices[title]
    
    # similarity score between wine which you selected and the others wines in database
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # sort results by the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # show top 10 results with the highest similarity score
    sim_scores = sim_scores[1:11]

    # select indices
    wine_indices = [i[0] for i in sim_scores]

    # return selected wine and the recommendation results in the new dataframe
    rec1 = wine[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery']].iloc[wine_indices]

    frames = [wine[wine["title"] == title], rec1]
    recommendation = pd.concat(frames, keys=['x', 'y'])
    return recommendation


# To get recommendation, you have to use wine title which index is not higher than size of wine dataframe (for quantile = 80 it is 31019)

Recommendation = wine_recommendations('Château Palmer 2009  Margaux')
Recommendation


# I don't think these are the worthful recommendations. Let's try to fix it and create new column with metadata of wine bottles.


# Before it, we have to change price and points on strings
wine['price'] = wine['price'].astype(str) 
wine['points'] = wine['points'].astype(str)

# And remove duplicates in titles
wine['title'].value_counts()
wine[wine["title"] == "Charles Heidsieck NV Brut Réserve  (Champagne)"] # We have to be careful because some of the titles are the two or more different wine types with different prices or points

# The best way to do it is add to metadata title of the wine and the rest of the variables 


def metadata(x):
    return ''.join(x['country']) + '' + '' .join(x['title']) + ''.join(x['points']) + ' ' + x['price'] + ' ' + ''.join(x['province'] + ' ' + x['variety'])

wine['metadata'] = wine.apply(metadata, axis=1)
wine['metadata'].value_counts() # As expected, we have couple of duplicates of not only wine title but it price, points, etc. which have to be removed from dataset
wine = wine.drop_duplicates('metadata')
wine['metadata'].value_counts() # It's done.


# We can remove the title from metadata column
def metadata(x):
    return ''.join(x['country']) + ' ' + ''.join(x['points']) + ' ' + x['price'] + ' ' + ''.join(x['province'] + ' ' + x['variety'])


wine['metadata'] = wine.apply(metadata, axis=1)
wine['metadata'].head() # It looks fine, we have all information in one column


from sklearn.feature_extraction.text import CountVectorizer

# Once again we will remove from the column all english stop words and repeat previous steps
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(wine['metadata']) 

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)

wine = wine.reset_index()
indices = pd.Series(wine.index, index=wine['title'])
wine = wine[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery', 'metadata']]


# We base on different similary matrix so we have to rebuild our recommendation's def

def wine_recommendations(title, cosine_sim=cosine_sim):
    
    # fit index of the wine to the title
    idx = indices[title]
    
    # similarity score between wine which you selected and the others wines in database
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # sort results by the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # show top 10 results with the highest similarity score
    sim_scores = sim_scores[1:11]

    # select indices
    wine_indices = [i[0] for i in sim_scores]

    # return selected wine and the recommendation results in the new dataframe
    recommendation = wine[['title', 'country', 'description', 'points', 'price', 'province', 'variety', 'winery', 'metadata']].iloc[wine_indices]

    frames = [wine[wine["title"] == title], recommendation]
    recommendation = pd.concat(frames, keys=['x', 'y'])
    return recommendation

Recommendation = wine_recommendations('Château Palmer 2009  Margaux')
Recommendation

"""
Ok, as expected, we have much better recommendations but I think they are too accurate.
It's more like "show me the substitute of the wine" not "show me the similar wines".
We can try to use less metadas and add price segment instead price.

Price segment - what is it?
I don't want to have in dataframe prices such as 30.00 USD or 31.00 USD. I prefer to describe it as one of the segment, in this case it would be "Low-Price".
I know that for you it could be strange that 30 USD per wine bottle is Low-Price but we have only the best wines and the highest price is 2500 USD so...
"""

# Before we create price segments, we have to change type of price from string to float

wine['price'] = wine['price'].astype(float)
wine["price"].dtype


wine["price"].plot(kind='hist')

# We have a lot of "cheap" wine bottles and only a few of prestige. Let's create 8 price buckets.

pd.qcut(wine["price"], q=8)

# Above code suggests us range of each bucket. It would works if we want to have the same number of observations in every bucket (price segment) but we don't so let's modify it a little but base on it

# Some useful statistics
np.std(wine["price"])
wine["price"].mean()


"""  
So I decided to use these ranges:
    - Low-Price [0 - 30.0]
    - Value (30.0 - 55.0]
    - Standard (55.0 - 80.0]
    - Premium (80.0 - 120.0]
    - Super Premium (120.0 - 250.0]
    - Ultra Premium (250.0 - 500.0]
    - Prestige (500.0 - 1000.00]
    - Prestige Plus (1000.0 - 2500.0]
"""

# We will work on copy of price column
wine["price segment"] = wine["price"]


# Loop which change price on the right price segment
segment = []

for row in wine["price segment"]:
    if row < 30:
        segment.append('LowPrice')
    elif row < 55:
        segment.append('Value')
    elif row < 80:
        segment.append('Standard')
    elif row < 120:
        segment.append('Premium')
    elif row < 250:
        segment.append('SuperPremium')
    elif row < 500:
        segment.append('UltraPremium')
    elif row < 1000:
        segment.append('Prestige')
    elif row >= 1000:
        segment.append('PrestigePlus')
    else:
        segment.append('Error')
        
wine['price segment'] = segment


wine['price segment'].value_counts() # No Errors, great!



# We have to rebuild our metadata

def metadata(x):
    return ''.join(x['country']) + ' ' + ''.join(x['points']) + ' ' + x['price segment'] + ' ' + ''.join(x['province'] + ' ' + x['variety'])


wine['metadata'] = wine.apply(metadata, axis=1)

# Once again we have to create the new similarity matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(wine['metadata']) 

cosine_sim = cosine_similarity(count_matrix, count_matrix)

wine = wine.reset_index()
indices = pd.Series(wine.index, index=wine['title'])

wine = wine[['title', 'country', 'description', 'points', 'price', 'price segment', 'province', 'variety', 'winery', 'metadata']]

# New recommendation's def

def wine_recommendations(title, cosine_sim=cosine_sim):
    
    # fit index of the wine to the title
    idx = indices[title]
    
    # similarity score between wine which you selected and the others wines in database
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # sort results by the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # show top 10 results with the highest similarity score
    sim_scores = sim_scores[1:11]

    # select indices
    wine_indices = [i[0] for i in sim_scores]

    # return selected wine and the recommendation results in the new dataframe
    recommendation = wine[['title', 'country', 'description', 'points', 'price', 'price segment', 'province', 'variety', 'winery', 'metadata']].iloc[wine_indices]

    frames = [wine[wine["title"] == title], recommendation]
    recommendation = pd.concat(frames, keys=['x', 'y'])
    return recommendation

Recommendation = wine_recommendations('Château Palmer 2009  Margaux')
Recommendation


# Results are already good but we can try to limit metadata to the most important to have more varied recommendations. Let's modify the previous step.


def metadata(x):
    return ''.join(x['points']) + ' ' + x['price segment'] + ' ' + x['variety']

wine['metadata'] = wine.apply(metadata, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(wine['metadata']) 

cosine_sim = cosine_similarity(count_matrix, count_matrix)

wine = wine.reset_index()
indices = pd.Series(wine.index, index=wine['title'])

wine = wine[['title', 'country', 'description', 'points', 'price', 'price segment', 'province', 'variety', 'winery', 'metadata']]

Recommendation = wine_recommendations('Château Palmer 2009  Margaux')
Recommendation

""" Now I am happy with the recommendations which we created. As you see, Château Palmer 2009 Margaux is a really good wine (98 points) but the bottle is expensive (380.00 USD). Thanks to our engine, we found a cheaper alternative.
Château Léoville Poyferré 2010  Saint-Julien has the same number of points but the price is only 92.00 USD and both wines are from Bordeaux, France!
I hope you had fun and learnt a lot. Thank you!

Kamil Jankowski
Stay in touch: https://www.linkedin.com/in/kamil-jankowski-bb8b38165

""'

 