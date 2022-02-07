# 추천 시스템

**추천 시스템은 왜 뜨고 있을까?**

- 롱테일 문제를 해결할 수 있다.
- 고객, 기업 양측 모두에게 이익이 된다.

**추천시스템 종류** (데이터 알고리즘에 의한 추천은 3번부터)

1. 사용자 프로파일링 기반
2. Segment 기반
3. 상품 연관규칙 기반
4. CF(협업 필터링) 기반
5. CBF(컨텐츠 베이스 필터링) 기반
6. 딥러닝 기반



# CBF_장르 유사도 기반 영화 추천

![image-20220207154434595](C:\Users\BDH\AppData\Roaming\Typora\typora-user-images\image-20220207154434595.png)

Quantum theory - 태거(태그 다는 사람)가 장르를 디테일하게 정리 (액션이 많이 들어갔는데 로맨스 등 수천개의 키워드) / 영화 장르를 양자 이론만큼 쪼개 놨다. / - 넷플릭스

[넷플릭스 양자이론](https://headnheart.tistory.com/19)



**특정 영화에 대해 장르가 유사한 영화를 추천해주는 서비스를 기획해보자!**



# 1. 데이터 로드

```python
import pandas as pd  # 데이터 분석을 위한 라이브러리
import numpy as np   # 넘파이 : 숫자 계산

# 워닝(버전, 업데이트 문제, 에러는 아님) 코드는 실행되는데 정보. 귀찮을 때 워닝 끄기
import warnings
warnings.filterwarnings('ignore')

movies = pd.read_csv('tmdb_5000_movies.csv')

print(movies.shape)  # 행, 열 개수 파악
movies.head()
```

```
(4803, 20)
```

Out[13]:

|      |    budget |                                            genres |                                     homepage |     id |                                          keywords | original_language |                           original_title |                                          overview | popularity |                              production_companies |                              production_countries | release_date |    revenue | runtime |                                  spoken_languages |   status |                                        tagline |                                    title | vote_average | vote_count |
| ---: | --------: | ------------------------------------------------: | -------------------------------------------: | -----: | ------------------------------------------------: | ----------------: | ---------------------------------------: | ------------------------------------------------: | ---------: | ------------------------------------------------: | ------------------------------------------------: | -----------: | ---------: | ------: | ------------------------------------------------: | -------: | ---------------------------------------------: | ---------------------------------------: | -----------: | ---------: |
|    0 | 237000000 | [{"id": 28, "name": "Action"}, {"id": 12, "nam... |                  http://www.avatarmovie.com/ |  19995 | [{"id": 1463, "name": "culture clash"}, {"id":... |                en |                                   Avatar | In the 22nd century, a paraplegic Marine is di... | 150.437577 | [{"name": "Ingenious Film Partners", "id": 289... | [{"iso_3166_1": "US", "name": "United States o... |   2009-12-10 | 2787965087 |   162.0 | [{"iso_639_1": "en", "name": "English"}, {"iso... | Released |                    Enter the World of Pandora. |                                   Avatar |          7.2 |      11800 |
|    1 | 300000000 | [{"id": 12, "name": "Adventure"}, {"id": 14, "... | http://disney.go.com/disneypictures/pirates/ |    285 | [{"id": 270, "name": "ocean"}, {"id": 726, "na... |                en | Pirates of the Caribbean: At World's End | Captain Barbossa, long believed to be dead, ha... | 139.082615 | [{"name": "Walt Disney Pictures", "id": 2}, {"... | [{"iso_3166_1": "US", "name": "United States o... |   2007-05-19 |  961000000 |   169.0 |          [{"iso_639_1": "en", "name": "English"}] | Released | At the end of the world, the adventure begins. | Pirates of the Caribbean: At World's End |          6.9 |       4500 |
|    2 | 245000000 | [{"id": 28, "name": "Action"}, {"id": 12, "nam... |  http://www.sonypictures.com/movies/spectre/ | 206647 | [{"id": 470, "name": "spy"}, {"id": 818, "name... |                en |                                  Spectre | A cryptic message from Bond’s past sends him o... | 107.376788 | [{"name": "Columbia Pictures", "id": 5}, {"nam... | [{"iso_3166_1": "GB", "name": "United Kingdom"... |   2015-10-26 |  880674609 |   148.0 | [{"iso_639_1": "fr", "name": "Fran\u00e7ais"},... | Released |                          A Plan No One Escapes |                                  Spectre |          6.3 |       4466 |
|    3 | 250000000 | [{"id": 28, "name": "Action"}, {"id": 80, "nam... |           http://www.thedarkknightrises.com/ |  49026 | [{"id": 849, "name": "dc comics"}, {"id": 853,... |                en |                    The Dark Knight Rises | Following the death of District Attorney Harve... | 112.312950 | [{"name": "Legendary Pictures", "id": 923}, {"... | [{"iso_3166_1": "US", "name": "United States o... |   2012-07-16 | 1084939099 |   165.0 |          [{"iso_639_1": "en", "name": "English"}] | Released |                                The Legend Ends |                    The Dark Knight Rises |          7.6 |       9106 |
|    4 | 260000000 | [{"id": 28, "name": "Action"}, {"id": 12, "nam... |         http://movies.disney.com/john-carter |  49529 | [{"id": 818, "name": "based on novel"}, {"id":... |                en |                              John Carter | John Carter is a war-weary, former military ca... |  43.926995 |       [{"name": "Walt Disney Pictures", "id": 2}] | [{"iso_3166_1": "US", "name": "United States o... |   2012-03-07 |  284139100 |   132.0 |          [{"iso_639_1": "en", "name": "English"}] | Released |           Lost in our world, found in another. |                              John Carter |          6.1 |       2124 |

```python
# 필요한 컬럼만 추출
movies_df = movies[['id','title','genres','vote_average','vote_count','popularity','keywords','overview']]
movies_df.head(3)
```

|      |     id |                                    title |                                            genres | vote_average | vote_count | popularity |                                          keywords | overview                                          |
| ---: | -----: | ---------------------------------------: | ------------------------------------------------: | -----------: | ---------: | ---------: | ------------------------------------------------: | ------------------------------------------------- |
|    0 |  19995 |                                   Avatar | [{"id": 28, "name": "Action"}, {"id": 12, "nam... |          7.2 |      11800 | 150.437577 | [{"id": 1463, "name": "culture clash"}, {"id":... | In the 22nd century, a paraplegic Marine is di... |
|    1 |    285 | Pirates of the Caribbean: At World's End | [{"id": 12, "name": "Adventure"}, {"id": 14, "... |          6.9 |       4500 | 139.082615 | [{"id": 270, "name": "ocean"}, {"id": 726, "na... | Captain Barbossa, long believed to be dead, ha... |
|    2 | 206647 |                                  Spectre | [{"id": 28, "name": "Action"}, {"id": 12, "nam... |          6.3 |       4466 | 107.376788 | [{"id": 470, "name": "spy"}, {"id": 818, "name... | A cryptic message from Bond’s past sends him o... |

```python
movies_df['genres'][0]
```

```
'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
```

```python
type(movies_df['genres'][0])
# str 타입인 것을 확인할 수 있다.
```

``````
str
``````



# 2. 데이터 전처리

## 1) genres, keywords 컬럼들의 str 형태를 list 형태로 바꿔주기

```python
from ast import literal_eval   # 문자열 파싱 라이브러리
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)
```

```python
movies_df['genres'][0]
```

```
[{'id': 28, 'name': 'Action'},
 {'id': 12, 'name': 'Adventure'},
 {'id': 14, 'name': 'Fantasy'},
 {'id': 878, 'name': 'Science Fiction'}]
```

```python
type(movies_df['genres'][0])
# str타입에서 list타입으로 바뀐 것을 확인할 수 있다.
```

``````
list
``````



## 1. 

##### 리스트 안에 딕셔너리로 여러 개의 장르 키워드가 문자열로 저장된 형태 -> 파싱 필요

```python
movies_df['genres'][0]
```

```
'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# 리스트 내 딕셔너리 형태인데, 문자열로 되어 있음
```

```python
type(movies_df['genres'][0])
# str 타입인 것을 확인할 수 있다.
```

``````
str
``````

```python
from ast import literal_eval  # 문자열 파싱 라이브러리
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)
```

