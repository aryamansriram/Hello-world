import pandas as pd
import numpy as np
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)
useful=lens.iloc[:,[1,5,6]]
user_id=useful.iloc[:,1]
R=useful.iloc[:,2]
N=useful.iloc[:,0]
j=0
Dc={}
ratings={}

for i in user_id:
    ide=i
    
    while(ide==user_id[j]):
        ratings[N[j]]=R[j]
        j=j+1
    
    Dc[ide]=ratings 
    print(ide)
  
    
    

