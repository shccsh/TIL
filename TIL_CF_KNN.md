# CF_KNN

- 갖고 있는 데이터가 없을 때는 CBF
- 취향 데이터가 쌓이게 되면 CF
- 고객이 평가하지 않은 null 값을 채워주는 역할

**Collaborative Filtering (협업 필터링)**

아이템 기반 협업 필터링

이번에 개봉하는 그 영화 볼까 말까 궁금하다면?

"친구에게 물어봐라" / 단, 취향이 비슷한 친구에게 물어봐야 한다.



# 0. 모듈 삽입 및 데이터 로드

```python
import pandas as pd
import numpy as np

movies = pd.read_csv('./data_movie_lens/movies.csv')
ratings = pd.read_csv('./data_movie_lens/ratings.csv')

print(movies.shape)
print(ratings.shape)

# 9천여개 영화에 대해 사용자들(600여명)이 평가한 10만여개 평점 데이터
```

```
(9742, 3)
(100836, 4)
```

```python
# 영화 정보 데이터
print(movies.shape)
movies.head()
```

```
(9742, 3)
```

Out[2]:

|      | movieId |                              title |                                          genres |
| ---: | ------: | ---------------------------------: | ----------------------------------------------: |
|    0 |       1 |                   Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|    1 |       2 |                     Jumanji (1995) |                    Adventure\|Children\|Fantasy |
|    2 |       3 |            Grumpier Old Men (1995) |                                 Comedy\|Romance |
|    3 |       4 |           Waiting to Exhale (1995) |                          Comedy\|Drama\|Romance |
|    4 |       5 | Father of the Bride Part II (1995) |                                          Comedy |

```python
# 유저들의 영화 별 평점 데이터
print(ratings.shape)
ratings
```

```
(100836, 4)
```

Out[3]:

|        | userId | movieId | rating |  timestamp |
| -----: | -----: | ------: | -----: | ---------: |
|      0 |      1 |       1 |    4.0 |  964982703 |
|      1 |      1 |       3 |    4.0 |  964981247 |
|      2 |      1 |       6 |    4.0 |  964982224 |
|      3 |      1 |      47 |    5.0 |  964983815 |
|      4 |      1 |      50 |    5.0 |  964982931 |
|    ... |    ... |     ... |    ... |        ... |
| 100831 |    610 |  166534 |    4.0 | 1493848402 |
| 100832 |    610 |  168248 |    5.0 | 1493850091 |
| 100833 |    610 |  168250 |    5.0 | 1494273047 |
| 100834 |    610 |  168252 |    5.0 | 1493846352 |
| 100835 |    610 |  170875 |    3.0 | 1493846415 |

100836 rows × 4 columns

# 1. 사용자-아이템 평점 행렬로 변환

```python
# 필요한 컬럼만 추출
ratings = ratings[['userId', 'movieId', 'rating']]
ratings
```

|        | userId | movieId | rating |
| -----: | -----: | ------: | ------ |
|      0 |      1 |       1 | 4.0    |
|      1 |      1 |       3 | 4.0    |
|      2 |      1 |       6 | 4.0    |
|      3 |      1 |      47 | 5.0    |
|      4 |      1 |      50 | 5.0    |
|    ... |    ... |     ... | ...    |
| 100831 |    610 |  166534 | 4.0    |
| 100832 |    610 |  168248 | 5.0    |
| 100833 |    610 |  168250 | 5.0    |
| 100834 |    610 |  168252 | 5.0    |
| 100835 |    610 |  170875 | 3.0    |

100836 rows × 3 columns

```python
# pivot_table 메소드를 사용해서 행렬 변환
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')

print(ratings_matrix.shape)
ratings_matrix
```

```
(610, 9724)
```

Out[7]:

| movieId |    1 |    2 |    3 |    4 |    5 |    6 |    7 |    8 |    9 |   10 |  ... | 193565 | 193567 | 193571 | 193573 | 193579 | 193581 | 193583 | 193585 | 193587 | 193609 |
| ------: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: |
|  userId |      |      |      |      |      |      |      |      |      |      |      |        |        |        |        |        |        |        |        |        |        |
|       1 |  4.0 |  NaN |  4.0 |  NaN |  NaN |  4.0 |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|       2 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|       3 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|       4 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|       5 |  4.0 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|     ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |    ... |    ... |    ... |    ... |    ... |    ... |    ... |    ... |    ... |    ... |
|     606 |  2.5 |  NaN |  NaN |  NaN |  NaN |  NaN |  2.5 |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|     607 |  4.0 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|     608 |  2.5 |  2.0 |  2.0 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  4.0 |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|     609 |  3.0 |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  NaN |  4.0 |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |
|     610 |  5.0 |  NaN |  NaN |  NaN |  NaN |  5.0 |  NaN |  NaN |  NaN |  NaN |  ... |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |    NaN |

610 rows × 9724 columns

```python
# title 컬럼을 얻기 위해 movies와 조인 수행
rating_movies = pd.merge(ratings, movies, on='movieId')
rating_movies
```

|        | userId | movieId | rating |                            title | genres                                          |
| -----: | -----: | ------: | -----: | -------------------------------: | ----------------------------------------------- |
|      0 |      1 |       1 |    4.0 |                 Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|      1 |      5 |       1 |    4.0 |                 Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|      2 |      7 |       1 |    4.5 |                 Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|      3 |     15 |       1 |    2.5 |                 Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|      4 |     17 |       1 |    4.5 |                 Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|    ... |    ... |     ... |    ... |                              ... | ...                                             |
| 100831 |    610 |  160341 |    2.5 |                 Bloodmoon (1997) | Action\|Thriller                                |
| 100832 |    610 |  160527 |    4.5 | Sympathy for the Underdog (1971) | Action\|Crime\|Drama                            |
| 100833 |    610 |  160836 |    3.0 |                    Hazard (2005) | Action\|Drama\|Thriller                         |
| 100834 |    610 |  163937 |    3.5 |               Blair Witch (2016) | Horror\|Thriller                                |
| 100835 |    610 |  163981 |    3.5 |                        31 (2016) | Horror                                          |

100836 rows × 5 columns

```python
# columns='title'로 title 컬럼으로 pivot 수행
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
ratings_matrix
```

|  title | '71 (2014) | 'Hellboy': The Seeds of Creation (2004) | 'Round Midnight (1986) | 'Salem's Lot (2004) | 'Til There Was You (1997) | 'Tis the Season for Love (2015) | 'burbs, The (1989) | 'night Mother (1986) | (500) Days of Summer (2009) | *batteries not included (1987) |  ... | Zulu (2013) | [REC] (2007) | [REC]² (2009) | [REC]³ 3 Génesis (2012) | anohana: The Flower We Saw That Day - The Movie (2013) | eXistenZ (1999) | xXx (2002) | xXx: State of the Union (2005) | ¡Three Amigos! (1986) | À nous la liberté (Freedom for Us) (1931) |
| -----: | ---------: | --------------------------------------: | ---------------------: | ------------------: | ------------------------: | ------------------------------: | -----------------: | -------------------: | --------------------------: | -----------------------------: | ---: | ----------: | -----------: | ------------: | ----------------------: | -----------------------------------------------------: | --------------: | ---------: | -----------------------------: | --------------------: | ----------------------------------------: |
| userId |            |                                         |                        |                     |                           |                                 |                    |                      |                             |                                |      |             |              |               |                         |                                                        |                 |            |                                |                       |                                           |
|      1 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   4.0 |                                       NaN |
|      2 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|      3 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|      4 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|      5 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|    ... |        ... |                                     ... |                    ... |                 ... |                       ... |                             ... |                ... |                  ... |                         ... |                            ... |  ... |         ... |          ... |           ... |                     ... |                                                    ... |             ... |        ... |                            ... |                   ... |                                       ... |
|    606 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|    607 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|    608 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             4.5 |        3.5 |                            NaN |                   NaN |                                       NaN |
|    609 |        NaN |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         NaN |                            NaN |  ... |         NaN |          NaN |           NaN |                     NaN |                                                    NaN |             NaN |        NaN |                            NaN |                   NaN |                                       NaN |
|    610 |        4.0 |                                     NaN |                    NaN |                 NaN |                       NaN |                             NaN |                NaN |                  NaN |                         3.5 |                            NaN |  ... |         NaN |          4.0 |           3.5 |                     3.0 |                                                    NaN |             NaN |        2.0 |                            1.5 |                   NaN |                                       NaN |

610 rows × 9719 columns

```python
# NaN 값을 모두 0으로 변환
ratings_matrix = ratings_matrix.fillna(0)
ratings_matrix
```

|  title | '71 (2014) | 'Hellboy': The Seeds of Creation (2004) | 'Round Midnight (1986) | 'Salem's Lot (2004) | 'Til There Was You (1997) | 'Tis the Season for Love (2015) | 'burbs, The (1989) | 'night Mother (1986) | (500) Days of Summer (2009) | *batteries not included (1987) |  ... | Zulu (2013) | [REC] (2007) | [REC]² (2009) | [REC]³ 3 Génesis (2012) | anohana: The Flower We Saw That Day - The Movie (2013) | eXistenZ (1999) | xXx (2002) | xXx: State of the Union (2005) | ¡Three Amigos! (1986) | À nous la liberté (Freedom for Us) (1931) |
| -----: | ---------: | --------------------------------------: | ---------------------: | ------------------: | ------------------------: | ------------------------------: | -----------------: | -------------------: | --------------------------: | -----------------------------: | ---: | ----------: | -----------: | ------------: | ----------------------: | -----------------------------------------------------: | --------------: | ---------: | -----------------------------: | --------------------: | ----------------------------------------: |
| userId |            |                                         |                        |                     |                           |                                 |                    |                      |                             |                                |      |             |              |               |                         |                                                        |                 |            |                                |                       |                                           |
|      1 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   4.0 |                                       0.0 |
|      2 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|      3 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|      4 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|      5 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|    ... |        ... |                                     ... |                    ... |                 ... |                       ... |                             ... |                ... |                  ... |                         ... |                            ... |  ... |         ... |          ... |           ... |                     ... |                                                    ... |             ... |        ... |                            ... |                   ... |                                       ... |
|    606 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|    607 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|    608 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             4.5 |        3.5 |                            0.0 |                   0.0 |                                       0.0 |
|    609 |        0.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         0.0 |                            0.0 |  ... |         0.0 |          0.0 |           0.0 |                     0.0 |                                                    0.0 |             0.0 |        0.0 |                            0.0 |                   0.0 |                                       0.0 |
|    610 |        4.0 |                                     0.0 |                    0.0 |                 0.0 |                       0.0 |                             0.0 |                0.0 |                  0.0 |                         3.5 |                            0.0 |  ... |         0.0 |          4.0 |           3.5 |                     3.0 |                                                    0.0 |             0.0 |        2.0 |                            1.5 |                   0.0 |                                       0.0 |

610 rows × 9719 columns

-> 사용자 - 아이템 행렬이 만들어졌다.

## 1) 영화와 영화들 간 유사도 산출

```python
# 아이템-사용자 행렬로 transpose 한다.
ratings_matrix_T = ratings_matrix.transpose()  #전치 행렬

print(ratings_matrix_T.shape)
ratings_matrix_T.head(5)
```

```
(9719, 610)
```

Out[8]:

|                                  userId |    1 |    2 |    3 |    4 |    5 |    6 |    7 |    8 |    9 |   10 |  ... |  601 |  602 |  603 |  604 |  605 |  606 |  607 |  608 |  609 |  610 |
| --------------------------------------: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|                                   title |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|                              '71 (2014) |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  ... |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  4.0 |
| 'Hellboy': The Seeds of Creation (2004) |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  ... |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |
|                  'Round Midnight (1986) |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  ... |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |
|                     'Salem's Lot (2004) |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  ... |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |
|               'Til There Was You (1997) |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  ... |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |

5 rows × 610 columns

```python
# 영화와 영화들 간 코사인 유사도 산출
from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

# cosine_similarity() 로 반환된 넘파이 행렬을 영화명을 매핑하여 DataFrame으로 변환
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns,
                          columns=ratings_matrix.columns)

print(item_sim_df.shape)
item_sim_df.head(3)
```

```
(9719, 9719)
```

Out[9]:

|                                   title | '71 (2014) | 'Hellboy': The Seeds of Creation (2004) | 'Round Midnight (1986) | 'Salem's Lot (2004) | 'Til There Was You (1997) | 'Tis the Season for Love (2015) | 'burbs, The (1989) | 'night Mother (1986) | (500) Days of Summer (2009) | *batteries not included (1987) |  ... | Zulu (2013) | [REC] (2007) | [REC]² (2009) | [REC]³ 3 Génesis (2012) | anohana: The Flower We Saw That Day - The Movie (2013) | eXistenZ (1999) | xXx (2002) | xXx: State of the Union (2005) | ¡Three Amigos! (1986) | À nous la liberté (Freedom for Us) (1931) |
| --------------------------------------: | ---------: | --------------------------------------: | ---------------------: | ------------------: | ------------------------: | ------------------------------: | -----------------: | -------------------: | --------------------------: | -----------------------------: | ---: | ----------: | -----------: | ------------: | ----------------------: | -----------------------------------------------------: | --------------: | ---------: | -----------------------------: | --------------------: | ----------------------------------------: |
|                                   title |            |                                         |                        |                     |                           |                                 |                    |                      |                             |                                |      |             |              |               |                         |                                                        |                 |            |                                |                       |                                           |
|                              '71 (2014) |        1.0 |                                0.000000 |               0.000000 |                 0.0 |                       0.0 |                             0.0 |           0.000000 |                  0.0 |                    0.141653 |                            0.0 |  ... |         0.0 |     0.342055 |      0.543305 |                0.707107 |                                                    0.0 |             0.0 |   0.139431 |                       0.327327 |                   0.0 |                                       0.0 |
| 'Hellboy': The Seeds of Creation (2004) |        0.0 |                                1.000000 |               0.707107 |                 0.0 |                       0.0 |                             0.0 |           0.000000 |                  0.0 |                    0.000000 |                            0.0 |  ... |         0.0 |     0.000000 |      0.000000 |                0.000000 |                                                    0.0 |             0.0 |   0.000000 |                       0.000000 |                   0.0 |                                       0.0 |
|                  'Round Midnight (1986) |        0.0 |                                0.707107 |               1.000000 |                 0.0 |                       0.0 |                             0.0 |           0.176777 |                  0.0 |                    0.000000 |                            0.0 |  ... |         0.0 |     0.000000 |      0.000000 |                0.000000 |                                                    0.0 |             0.0 |   0.000000 |                       0.000000 |                   0.0 |                                       0.0 |

3 rows × 9719 columns

```python
# Godfather와 유사한 영화 6개 확인해보기
item_sim_df["Godfather, The (1972)"].sort_values(ascending=False)[:6]
```

```
title
Godfather, The (1972)                        1.000000
Godfather: Part II, The (1974)               0.821773
Goodfellas (1990)                            0.664841
One Flew Over the Cuckoo's Nest (1975)       0.620536
Star Wars: Episode IV - A New Hope (1977)    0.595317
Fargo (1996)                                 0.588614
Name: Godfather, The (1972), dtype: float64
```

```python
# 자기 것 빼고 인셉션과 유사한 영화 5개 확인해보기
item_sim_df["Inception (2010)"].sort_values(ascending=False)[1:6]
```

```
title
Dark Knight, The (2008)          0.727263
Inglourious Basterds (2009)      0.646103
Shutter Island (2010)            0.617736
Dark Knight Rises, The (2012)    0.617504
Fight Club (1999)                0.615417
Name: Inception (2010), dtype: float64
```

## 2) 아이템 기반 인접 이웃 협업 필터링으로 개인화된 영화 추천

```python
# 평점 벡터(행 벡터)와 유사도 벡터(열 벡터)를 내적(dot)해서 예측 평점을 계산하는 함수 정의
def predict_rating(ratings_arr, item_sim_arr):
    ratings_pred = ratings_arr.dot(item_sim_arr)/ np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred
```

```python
ratings_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
ratings_pred
```

```
array([[0.07034471, 0.5778545 , 0.32169559, ..., 0.13602448, 0.29295452,
        0.72034722],
       [0.01826008, 0.04274424, 0.01886104, ..., 0.02452792, 0.01756305,
        0.        ],
       [0.01188449, 0.03027871, 0.06443729, ..., 0.00922874, 0.01041982,
        0.08450144],
       ...,
       [0.32443466, 1.02254119, 0.5984666 , ..., 0.53858621, 0.52763888,
        0.69887065],
       [0.00483488, 0.05359271, 0.02625119, ..., 0.0131077 , 0.01832826,
        0.03337679],
       [3.62830323, 1.51791811, 0.83366781, ..., 1.81609065, 0.56507537,
        0.57465402]])
```

```python
# 데이터프레임으로 변환
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index,
                                   columns = ratings_matrix.columns)
print(ratings_pred_matrix.shape)
ratings_pred_matrix.head(10)
```

```
(610, 9719)
```

Out[15]:

|  title | '71 (2014) | 'Hellboy': The Seeds of Creation (2004) | 'Round Midnight (1986) | 'Salem's Lot (2004) | 'Til There Was You (1997) | 'Tis the Season for Love (2015) | 'burbs, The (1989) | 'night Mother (1986) | (500) Days of Summer (2009) | *batteries not included (1987) |  ... | Zulu (2013) | [REC] (2007) | [REC]² (2009) | [REC]³ 3 Génesis (2012) | anohana: The Flower We Saw That Day - The Movie (2013) | eXistenZ (1999) | xXx (2002) | xXx: State of the Union (2005) | ¡Three Amigos! (1986) | À nous la liberté (Freedom for Us) (1931) |
| -----: | ---------: | --------------------------------------: | ---------------------: | ------------------: | ------------------------: | ------------------------------: | -----------------: | -------------------: | --------------------------: | -----------------------------: | ---: | ----------: | -----------: | ------------: | ----------------------: | -----------------------------------------------------: | --------------: | ---------: | -----------------------------: | --------------------: | ----------------------------------------: |
| userId |            |                                         |                        |                     |                           |                                 |                    |                      |                             |                                |      |             |              |               |                         |                                                        |                 |            |                                |                       |                                           |
|      1 |   0.070345 |                                0.577855 |               0.321696 |            0.227055 |                  0.206958 |                        0.194615 |           0.249883 |             0.102542 |                    0.157084 |                       0.178197 |  ... |    0.113608 |     0.181738 |      0.133962 |                0.128574 |                                               0.006179 |        0.212070 |   0.192921 |                       0.136024 |              0.292955 |                                  0.720347 |
|      2 |   0.018260 |                                0.042744 |               0.018861 |            0.000000 |                  0.000000 |                        0.035995 |           0.013413 |             0.002314 |                    0.032213 |                       0.014863 |  ... |    0.015640 |     0.020855 |      0.020119 |                0.015745 |                                               0.049983 |        0.014876 |   0.021616 |                       0.024528 |              0.017563 |                                  0.000000 |
|      3 |   0.011884 |                                0.030279 |               0.064437 |            0.003762 |                  0.003749 |                        0.002722 |           0.014625 |             0.002085 |                    0.005666 |                       0.006272 |  ... |    0.006923 |     0.011665 |      0.011800 |                0.012225 |                                               0.000000 |        0.008194 |   0.007017 |                       0.009229 |              0.010420 |                                  0.084501 |
|      4 |   0.049145 |                                0.277628 |               0.160448 |            0.206892 |                  0.309632 |                        0.042337 |           0.130048 |             0.116442 |                    0.099785 |                       0.097432 |  ... |    0.051269 |     0.076051 |      0.055563 |                0.054137 |                                               0.008343 |        0.159242 |   0.100941 |                       0.062253 |              0.146054 |                                  0.231187 |
|      5 |   0.007278 |                                0.066951 |               0.041879 |            0.013880 |                  0.024842 |                        0.018240 |           0.026405 |             0.018673 |                    0.021591 |                       0.018841 |  ... |    0.009689 |     0.022246 |      0.013360 |                0.012378 |                                               0.000000 |        0.025839 |   0.023712 |                       0.018012 |              0.028133 |                                  0.052315 |
|      6 |   0.022967 |                                0.122637 |               0.071967 |            0.188898 |                  0.222312 |                        0.071049 |           0.164005 |             0.076842 |                    0.095137 |                       0.122987 |  ... |    0.047859 |     0.102881 |      0.062647 |                0.060845 |                                               0.000000 |        0.118870 |   0.120876 |                       0.080545 |              0.152925 |                                  0.181533 |
|      7 |   0.062503 |                                0.372868 |               0.198837 |            0.034989 |                  0.046235 |                        0.209330 |           0.095166 |             0.054397 |                    0.118785 |                       0.080396 |  ... |    0.049927 |     0.117622 |      0.092001 |                0.089025 |                                               0.017309 |        0.102175 |   0.136675 |                       0.127286 |              0.098902 |                                  0.141889 |
|      8 |   0.009577 |                                0.084303 |               0.047613 |            0.027602 |                  0.043137 |                        0.024943 |           0.031655 |             0.014261 |                    0.026341 |                       0.024008 |  ... |    0.011018 |     0.026196 |      0.019240 |                0.015344 |                                               0.000000 |        0.027063 |   0.030581 |                       0.024177 |              0.033878 |                                  0.075097 |
|      9 |   0.016342 |                                0.081805 |               0.043044 |            0.039426 |                  0.026811 |                        0.049085 |           0.027855 |             0.014940 |                    0.023836 |                       0.024325 |  ... |    0.012365 |     0.022456 |      0.016859 |                0.017366 |                                               0.000000 |        0.029529 |   0.032399 |                       0.021871 |              0.027114 |                                  0.029839 |
|     10 |   0.044189 |                                0.155954 |               0.075501 |            0.106136 |                  0.066934 |                        0.253253 |           0.044313 |             0.020359 |                    0.117624 |                       0.040900 |  ... |    0.046781 |     0.069744 |      0.046964 |                0.045225 |                                               0.041817 |        0.041142 |   0.083420 |                       0.081787 |              0.055053 |                                  0.019026 |

10 rows × 9719 columns

-> 영화 별 예측 평점이 나오게 된다.

## 3) 예측 평점 정확도를 판단하기 위해 오차 함수인 MSE 이용

```python
from sklearn.metrics import mean_squared_error

# 사용자가 평점을 부여한 영화에 대해서만 예측 성능 평가 MSE 를 구함. 
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print('아이템 기반 모든 인접 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values ))
```

```
아이템 기반 모든 인접 이웃 MSE:  9.895354759094706
```

## 4) top_n 유사도를 가진 데이터들에 대해서만 예측 평점 계산

```python
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 열 크기만큼 Loop 수행. 
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개 데이터 행렬의 index 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        # 개인화된 예측 평점을 계산
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row, :][top_n_items].T) 
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))        
    return pred
```

```python
ratings_pred = predict_rating_topsim(ratings_matrix.values , item_sim_df.values, n=20)
print('아이템 기반 인접 TOP-20 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values ))

# 계산된 예측 평점 데이터는 DataFrame으로 재생성
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index,
                                   columns = ratings_matrix.columns)
```

```
아이템 기반 인접 TOP-20 이웃 MSE:  3.695009387428144
```

-> 최종적인 영화 별 예측 평점 데이터가 만들어졌다.

# 2. 사용자에게 영화 추천을 해보자

```python
# 사용자 9번에게 영화를 추천해보자
# 추천에 앞서 9번 사용자가 높은 평점을 준 영화를 확인해보면
user_rating_id = ratings_matrix.loc[9, :]
user_rating_id[ user_rating_id > 0].sort_values(ascending=False)[:10]
```

```
title
Adaptation (2002)                                                                 5.0
Citizen Kane (1941)                                                               5.0
Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)    5.0
Producers, The (1968)                                                             5.0
Lord of the Rings: The Two Towers, The (2002)                                     5.0
Lord of the Rings: The Fellowship of the Ring, The (2001)                         5.0
Back to the Future (1985)                                                         5.0
Austin Powers in Goldmember (2002)                                                5.0
Minority Report (2002)                                                            4.0
Witness (1985)                                                                    4.0
Name: 9, dtype: float64
```

- 오스틴 파워, 반지의 제왕 등 흥행성이 높은 영화에 높은 평점을 주고 있다.
- 대작 영화, 어드벤처, 코미디 등의 영화를 좋아할 것이라고 예상할 수 있다.

## 1) 사용자가 관람하지 않은 영화 중에서 영화를 추천해보자

**user_rating이 0보다 크면 기존에 관람한 영화라는 점을 이용해서 계산**

```python
def get_unseen_movies(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 영화정보 추출하여 Series로 반환함
    # 반환된 user_rating은 영화명(title)을 index로 가지는 Series 객체임
    user_rating = ratings_matrix.loc[userId, :]
    
    # user_rating이 0보다 크면 기존에 관람한 영화임. 대상 index를 추출하여 list 객체로 만듬
    already_seen = user_rating[ user_rating > 0].index.tolist()
    
    # 모든 영화명을 list 객체로 만듬
    movies_list = ratings_matrix.columns.tolist()
    
    # list comprehension으로 already_seen에 해당하는 movie는 movies_list에서 제외함. 
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]
    
    return unseen_list
```

```python
# pred_df : 앞서 계산된 영화 별 예측 평점
# unseen_list : 사용자가 보지 않은 영화들
# top_n : 상위 n개를 가져온다.

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    # 예측 평점 DataFrame에서 사용자id index와 unseen_list로 들어온 영화명 컬럼을 추출하여
    # 가장 예측 평점이 높은 순으로 정렬함. 
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies
```

```python
# 사용자가 관람하지 않는 영화명 추출   
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천 
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

# 평점 데이타를 DataFrame으로 생성. 
recomm_movies = pd.DataFrame(data=recomm_movies.values,index=recomm_movies.index, columns=['pred_score'])
recomm_movies
```

|                                                              | pred_score |
| -----------------------------------------------------------: | ---------: |
|                                                        title |            |
|                                                 Shrek (2001) |   0.866202 |
|                                            Spider-Man (2002) |   0.857854 |
|                                     Last Samurai, The (2003) |   0.817473 |
|                  Indiana Jones and the Temple of Doom (1984) |   0.816626 |
|                                  Matrix Reloaded, The (2003) |   0.800990 |
| Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) |   0.765159 |
|                                             Gladiator (2000) |   0.740956 |
|                                           Matrix, The (1999) |   0.732693 |
| Pirates of the Caribbean: The Curse of the Black Pearl (2003) |   0.689591 |
|        Lord of the Rings: The Return of the King, The (2003) |   0.676711 |

- 결론 :

  아이템 기반의 인접 이웃 협업 필터링으로

  사용자의 영화 예측 평점을 계산해서

  상위 10개의 영화를 추천해주었다.