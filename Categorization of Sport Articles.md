```python
# Import required libraries
# Read data
# Explore and pre-process text
# Topic Modeling
```


```python
import numpy as np
import pandas as pd 
import nltk 
import spacy 
import re 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 

```


```python
#reading the data 
```


```python
file = open(r'C:\Users\Dell\Desktop\Scalable systems\nlp\categorization of sports articles project\bbc_sports_articles\006.txt', mode='rt', encoding='utf-8')
text=file.read()
file.close()

```


```python
text
```




    'Isinbayeva claims new world best\n\nPole vaulter Yelena Isinbayeva broke her own indoor world record by clearing 4.89 metres in Lievin on Saturday.\n\nIt was the Russian\'s 12th world record of her career and came just a few days after she cleared 4.88m at the Norwich Union Grand Prix in Birmingham. The Olympic champion went on to attempt 5.05m at the meeting on France but failed to clear that height. In the men\'s 60m, former Olympic 100m champion Maurice Greene could only finish second to Leonard Scott. It was Greene\'s second consecutive defeat at the hands of his fellow American, who also won in Birmingham last week. "I ran my race perfectly," said Scott, who won in 6.46secs, his best time indoors. "I am happy even if I know that Maurice is a long way from being at his peak at the start of the season."\n'




```python
# reading the file names 
```


```python
file_names =os.listdir(r'C:\Users\Dell\Desktop\Scalable systems\nlp\categorization of sports articles project\bbc_sports_articles') 
```


```python
# printing the name of files
file_names[:10]
```




    ['001.txt',
     '002.txt',
     '003.txt',
     '004.txt',
     '005.txt',
     '006.txt',
     '007.txt',
     '008.txt',
     '009.txt',
     '010.txt']




```python
# appending the content of every text file into a list 
```


```python
articles=[]

for f in file_names:
    file = open(r"C:\Users\Dell\Desktop\Scalable systems\nlp\categorization of sports articles project\bbc_sports_articles/" +f, mode="rt", encoding='utf-8')
    text=file.read()
    file.close()
    articles.append(text)
    
```


```python
articles[10]
```




    'Radcliffe yet to answer GB call\n\nPaula Radcliffe has been granted extra time to decide whether to compete in the World Cross-Country Championships.\n\nThe 31-year-old is concerned the event, which starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. "There is no question that Paula would be a huge asset to the GB team," said Zara Hyde Peters of UK Athletics. "But she is working out whether she can accommodate the worlds without too much compromise in her marathon training." Radcliffe must make a decision by Tuesday - the deadline for team nominations. British team member Hayley Yelling said the team would understand if Radcliffe opted out of the event. "It would be fantastic to have Paula in the team," said the European cross-country champion. "But you have to remember that athletics is basically an individual sport and anything achieved for the team is a bonus. "She is not messing us around. We all understand the problem." Radcliffe was world cross-country champion in 2001 and 2002 but missed last year\'s event because of injury. In her absence, the GB team won bronze in Brussels.\n'




```python
# number of the textfiles 
len(articles)
```




    471




```python
# text cleaning and pre processing 
```


```python
clean_articles=[]
for i in articles:
    clean_articles.append(i.replace("\n"," ").replace("\'"," "))
```


```python
clean_articles[10]
```




    'Radcliffe yet to answer GB call  Paula Radcliffe has been granted extra time to decide whether to compete in the World Cross-Country Championships.  The 31-year-old is concerned the event, which starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. "There is no question that Paula would be a huge asset to the GB team," said Zara Hyde Peters of UK Athletics. "But she is working out whether she can accommodate the worlds without too much compromise in her marathon training." Radcliffe must make a decision by Tuesday - the deadline for team nominations. British team member Hayley Yelling said the team would understand if Radcliffe opted out of the event. "It would be fantastic to have Paula in the team," said the European cross-country champion. "But you have to remember that athletics is basically an individual sport and anything achieved for the team is a bonus. "She is not messing us around. We all understand the problem." Radcliffe was world cross-country champion in 2001 and 2002 but missed last year s event because of injury. In her absence, the GB team won bronze in Brussels. '




```python
# remvoing special characters, numbers and punctuations, only keeping alphabets 
```


```python
clean_articles = [re.sub("[^a-zA-z]", " ",x) for x in clean_articles]
```


```python
clean_articles[10]
```




    'Radcliffe yet to answer GB call  Paula Radcliffe has been granted extra time to decide whether to compete in the World Cross Country Championships   The    year old is concerned the event  which starts on    March in France  could upset her preparations for the London Marathon on    April   There is no question that Paula would be a huge asset to the GB team   said Zara Hyde Peters of UK Athletics   But she is working out whether she can accommodate the worlds without too much compromise in her marathon training   Radcliffe must make a decision by Tuesday   the deadline for team nominations  British team member Hayley Yelling said the team would understand if Radcliffe opted out of the event   It would be fantastic to have Paula in the team   said the European cross country champion   But you have to remember that athletics is basically an individual sport and anything achieved for the team is a bonus   She is not messing us around  We all understand the problem   Radcliffe was world cross country champion in      and      but missed last year s event because of injury  In her absence  the GB team won bronze in Brussels  '




```python
# removing the single character terms
```


```python
clean_articles =[' '.join([w for w in x.split() if len(w)>1]) for x in clean_articles]
```


```python
clean_articles[10]
```




    'Radcliffe yet to answer GB call Paula Radcliffe has been granted extra time to decide whether to compete in the World Cross Country Championships The year old is concerned the event which starts on March in France could upset her preparations for the London Marathon on April There is no question that Paula would be huge asset to the GB team said Zara Hyde Peters of UK Athletics But she is working out whether she can accommodate the worlds without too much compromise in her marathon training Radcliffe must make decision by Tuesday the deadline for team nominations British team member Hayley Yelling said the team would understand if Radcliffe opted out of the event It would be fantastic to have Paula in the team said the European cross country champion But you have to remember that athletics is basically an individual sport and anything achieved for the team is bonus She is not messing us around We all understand the problem Radcliffe was world cross country champion in and but missed last year event because of injury In her absence the GB team won bronze in Brussels'




```python
# lower casing every article 
```


```python
clean_articles =[x.lower() for x in clean_articles]
```


```python
clean_articles[10]
```




    'radcliffe yet to answer gb call paula radcliffe has been granted extra time to decide whether to compete in the world cross country championships the year old is concerned the event which starts on march in france could upset her preparations for the london marathon on april there is no question that paula would be huge asset to the gb team said zara hyde peters of uk athletics but she is working out whether she can accommodate the worlds without too much compromise in her marathon training radcliffe must make decision by tuesday the deadline for team nominations british team member hayley yelling said the team would understand if radcliffe opted out of the event it would be fantastic to have paula in the team said the european cross country champion but you have to remember that athletics is basically an individual sport and anything achieved for the team is bonus she is not messing us around we all understand the problem radcliffe was world cross country champion in and but missed last year event because of injury in her absence the gb team won bronze in brussels'




```python
# checking most frequent words 
```


```python
# checking for top 30 words
def freqDist(x, terms=30):
    text= " ".join(text for text in x)
    words = text.split()
    
    # developing the dictionary with key as a word and value as its "counts of appearence in the words list above"
    fdist = nltk.FreqDist(words)
    word_df = pd.DataFrame({"word": list(fdist.keys()), "count":list(fdist.values())})
    # selecting the top n words 
    d = word_df.nlargest(columns="count", n=terms)
                            
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel="Count")
    plt.show()
```


```python
freqDist(clean_articles)
```


![png](output_26_0.png)



```python
# doing lemmatization and removing stop words 
```


```python
nlp =spacy.load("en_core_web_sm")
nlp.max_length = 9900000
```


```python
#lemmatization 
clean_articles = [" ".join(token.lemma_ for token in nlp(x)) for x in clean_articles]

clean_articles = [" ".join(w for w in x.split() if nlp.vocab[w].is_stop==False) for x in clean_articles]
```


```python
clean_articles[10]
```




    'radcliffe answer gb paula radcliffe grant extra time decide compete world cross country championship year old concern event start march france upset -PRON- preparation london marathon april question paula huge asset gb team zara hyde peters uk athletics -PRON- work -PRON- accommodate world compromise -PRON- marathon training radcliffe decision tuesday deadline team nomination british team member hayley yelling team understand radcliffe opt event -PRON- fantastic paula team european cross country champion -PRON- remember athletic basically individual sport achieve team bonus -PRON- mess -PRON- -PRON- understand problem radcliffe world cross country champion miss year event injury -PRON- absence gb team win bronze brussels'




```python
freqDist(clean_articles)
```


![png](output_31_0.png)



```python
# removing -PRON- 
```


```python
clean_articles = [re.sub("-PRON-"," ",x) for x in clean_articles]
```


```python
freqDist(clean_articles)
```


![png](output_34_0.png)



```python
## Topic Modelling LSA , LDA and Non Negative Matrix Factorization
```


```python
# Topic Modelling using LSA 
```


```python
# Finding how many unique terms are there
```


```python
def get_words(x):
    text = " ".join(text for text in x)
    
    return set(text.split())
```


```python
unique_words = get_words(clean_articles)
```


```python
len(unique_words)
```




    7772




```python
# creating document matrix using Tfid vectorization from sklearn 
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer  
```


```python
vectorizer= TfidfVectorizer()
X = vectorizer.fit_transform(clean_articles)
X.shape
```




    (471, 7768)




```python
len(vectorizer.get_feature_names())
```




    7768




```python
X_df = pd.DataFrame.sparse.from_spmatrix(X, columns = vectorizer.get_feature_names(), index = range(len(clean_articles)))
```


```python
X_df.shape
```




    (471, 7768)




```python
# viewing our document matrix 
```


```python
X_df.iloc[:10,100:120]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>administrative</th>
      <th>administrator</th>
      <th>admirable</th>
      <th>admiration</th>
      <th>admire</th>
      <th>admired</th>
      <th>admissibility</th>
      <th>admission</th>
      <th>admit</th>
      <th>admittedly</th>
      <th>ado</th>
      <th>adrian</th>
      <th>adrift</th>
      <th>advance</th>
      <th>advanced</th>
      <th>advancinhg</th>
      <th>advanta</th>
      <th>advantage</th>
      <th>adventure</th>
      <th>adversity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.043121</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# decreasing the sparsity of the document matrix 
```


```python
vectorizer = TfidfVectorizer(max_features = 1000, min_df =5, max_df=0.9)
X = vectorizer.fit_transform(clean_articles)
X.shape

```




    (471, 1000)




```python
# reducing the dimensionality of the document matrix to a smaller number(number of topics)
```


```python
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components = 4, random_state=12, n_iter=100)
```


```python
svd_model.fit(X)
```




    TruncatedSVD(algorithm='randomized', n_components=4, n_iter=100,
                 random_state=12, tol=0.0)




```python
svd_model.components_.shape
```




    (4, 1000)




```python

```


```python
# top 20 words in each of the four topics 
```


```python
terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    term_comp = zip(terms, comp)
    sorted_terms = sorted(term_comp, key = lambda x:x[1], reverse=True)[:70]
    
    print("Topic "+ str(i)+ ":")
    topics=[]
    for item in sorted_terms:
        topics.append(item[0])
    
    print(topics)
    print("\n")
```

    Topic 0:
    ['england', 'win', 'play', 'year', 'game', 'test', 'match', 'world', 'good', 'team', 'player', 'ireland', 'final', 'time', 'day', 'new', 'cricket', 'rugby', 'second', 'come', 'coach', 'france', 'set', 'open', 'injury', 'cup', 'run', 'think', 'tour', 'half', 'south', 'australia', 'international', 'series', 'start', 'nation', 'victory', 'beat', 'try', 'champion', 'week', 'captain', 'great', 'season', 'ball', 'wale', 'robinson', 'lose', 've', 'zealand', 'title', 'pakistan', 'africa', 'seed', 'jones', 'add', 'williams', 'look', 'scotland', 'want', 'old', 'race', 'squad', 'india', 'way', 'number', 'break', 'know', 'australian', 'wicket']
    
    
    Topic 1:
    ['win', 'seed', 'champion', 'indoor', 'open', 'title', 'world', 'olympic', 'race', 'final', 'set', 'year', 'european', 'roddick', 'woman', 'round', 'event', 'federer', 'compete', 'hewitt', 'beat', 'break', 'record', 'britain', 'championship', 'grand', 'birmingham', 'australian', 'gold', 'american', 'agassi', 'athens', 'athlete', 'jump', 'british', 'second', 'medal', 'davenport', 'holmes', 'double', 'tennis', 'henman', 'madrid', 'safin', 'slam', 'wimbledon', 'old', 'francis', 'marathon', 'cross', 'holme', 'russian', 'moya', 'season', 'number', 'davis', 'radcliffe', 'serve', 'gardener', 'kelly', 'spain', 'good', 'semi', 'tournament', 'feel', 'great', 'drug', 'relay', 'greene', 'personal']
    
    
    Topic 2:
    ['england', 'ireland', 'robinson', 'france', 'nation', 'wale', 'rugby', 'half', 'scotland', 'wales', 'italy', 'try', 'coach', 'game', 'penalty', 'williams', 'bath', 'leicester', 'wilkinson', 'kick', 'centre', 'hodgson', 'gara', 'driscoll', 'player', 'lion', 'irish', 'henson', 'injury', 'ruddock', 'andy', 'barkley', 'stade', 'referee', 'laporte', 'lewsey', 'scrum', 'fly', 'wing', 'thomas', 'cueto', 'flanker', 'sale', 'wasp', 'cardiff', 'twickenham', 'woodward', 'welsh', 'gloucester', 'slam', 'jonny', 'saturday', 'goal', 'tait', 'prop', 'cup', 'line', 'sullivan', 'dawson', 'newcastle', 'kaplan', 'mike', 'matt', 'replacement', 'charlie', 'squad', 'dublin', 'corry', 'lock', 'moody']
    
    
    Topic 3:
    ['kenteris', 'greek', 'thanou', 'iaaf', 'drug', 'athens', 'olympic', 'athlete', 'charge', 'ban', 'dope', 'olympics', 'sprinter', 'england', 'miss', 'indoor', 'test', 'tribunal', 'race', 'decision', 'athletic', 'european', 'sport', 'medal', 'robinson', 'trial', 'federation', 'ireland', 'athletics', 'rugby', 'evidence', 'marathon', 'suspend', 'gold', 'conte', 'compete', 'birmingham', 'radcliffe', 'kostas', 'coach', 'balco', 'pair', 'record', 'nation', 'holmes', 'case', 'training', 'cross', 'lewis', 'appeal', 'francis', 'committee', 'holme', 'fail', 'hearing', 'britain', 'kelly', 'london', 'clear', 'jones', 'jump', 'avoid', 'official', 'body', 'silver', 'anti', 'international', 'madrid', 'johnson', 'wale']
    
    
    


```python
# catergorization of articles
```


```python
lsa_topic_matrix = svd_model.transform(X)
lsa_topic_matrix.shape
```




    (471, 4)




```python
lsa_topic_matrix[32]
```




    array([ 0.16812655,  0.10387154, -0.30388207,  0.48508372])




```python
np.argmax(lsa_topic_matrix[32])
```




    3




```python
# number of articles belonging to each categories 
```


```python
# count for topic 0
count_0=0
ls_0=[]
# count for topic 1
count_1=0
ls_1=[]
# count for topic 2
count_2=0
ls_2=[]
# count for topic 3
count_3=0
ls_3=[]
for i in range(lsa_topic_matrix.shape[0]):
    if np.argmax(lsa_topic_matrix[i])==0:
        count_0+=1
        ls_0.append(i+1)
    elif np.argmax(lsa_topic_matrix[i])==1:
        count_1+=1
        ls_1.append(i+1)
    elif np.argmax(lsa_topic_matrix[i])==2:
        count_2+=1
        ls_2.append(i+1)
    else:
        count_3+=1
        ls_3.append(i+1)
print("Topic 0 (Cricket): " + str(count_0)+ "  " , "Topic 1 (Tennis): " + str(count_1)+"  ", "Topic 2(Football): " + str(count_2)+"  ", "Topic 3(athletics): " + str(count_3))
print()
print("Topic 0: ", ls_0)
print()
print("Topic 1: ", ls_1)
print()
print("Topic 2: ", ls_2)
print()
print("Topic 3: ", ls_3)
```

    Topic 0 (Cricket): 425   Topic 1 (Tennis): 17   Topic 2(Football): 3   Topic 3(athletics): 26
    
    Topic 0:  [1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 14, 15, 18, 21, 22, 24, 28, 30, 31, 34, 37, 38, 40, 41, 42, 43, 52, 55, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 83, 84, 85, 86, 87, 90, 92, 93, 94, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471]
    
    Topic 1:  [10, 12, 20, 25, 27, 29, 32, 53, 56, 63, 73, 95, 96, 99, 379, 420, 440]
    
    Topic 2:  [268, 296, 299]
    
    Topic 3:  [4, 16, 17, 19, 23, 26, 33, 35, 36, 39, 44, 45, 46, 47, 48, 49, 50, 51, 54, 79, 80, 81, 82, 88, 89, 91]
    


```python
# Using the Non Negative Matrix Factorization method 
```


```python
from sklearn.decomposition import NMF
```


```python
nmf = NMF(n_components=4, random_state=1, 
          alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
```


```python
nmf.fit(X)
```




    NMF(alpha=0.1, beta_loss='frobenius', init='nndsvd', l1_ratio=0.5, max_iter=200,
        n_components=4, random_state=1, shuffle=False, solver='cd', tol=0.0001,
        verbose=0)




```python
nmf.components_.shape
```




    (4, 1000)




```python
terms = vectorizer.get_feature_names()

for i, comp in enumerate(nmf.components_):
    term_comp = zip(terms, comp)
    sorted_terms = sorted(term_comp, key = lambda x:x[1], reverse=True)[:70]
    
    print("Topic "+ str(i)+ ":")
    topics=[]
    for item in sorted_terms:
        topics.append(item[0])
    
    print(topics)
    print("\n")
```

    Topic 0:
    ['win', 'open', 'final', 'year', 'seed', 'world', 'set', 'champion', 'title', 'indoor', 'race', 'second', 'good', 'beat', 'roddick', 'play', 'australian', 'olympic', 'break', 'round', 'european', 'woman', 'time', 'match', 'federer', 'event', 'record', 'hewitt', 'agassi', 'number', 'old', 'championship', 'season', 'feel', 'victory', 'american', 'grand', 'come', 'cup', 'britain', 'great', 'tennis', 'compete', 'british', 'run', 'henman', 'davenport', 'start', 'double', 'think', 'injury', 'lose', 'wimbledon', 'safin', 'man', 'serve', 'game', 'slam', 'week', 'jump', 'gold', 'birmingham', 'davis', 'tournament', 'moya', 'medal', 'cross', 'fourth', 'like', 'russian']
    
    
    Topic 1:
    ['test', 'cricket', 'pakistan', 'india', 'wicket', 'series', 'day', 'australia', 'south', 'tour', 'play', 'ball', 'england', 'africa', 'run', 'match', 'sri', 'vaughan', 'team', 'batsman', 'zimbabwe', 'new', 'catch', 'captain', 'mohammad', 'zealand', 'bowl', 'bowler', 'khan', 'bangladesh', 'international', 'michael', 'wkt', 'shoaib', 'board', 'capt', 'spinner', 'lanka', 'xi', 'home', 'strauss', 'ponting', 'inzamam', 'andrew', 'boundary', 'bat', 'boje', 'score', 'jones', 'flintoff', 'century', 'trescothick', 'game', 'smith', 'west', 'ul', 'good', 'langer', 'gillespie', 'need', 'player', 'icc', 'warne', 'innings', 'come', 'ntini', 'kallis', 'ganguly', 'gilchrist', 'hayden']
    
    
    Topic 2:
    ['england', 'ireland', 'rugby', 'robinson', 'france', 'nation', 'wale', 'half', 'game', 'coach', 'scotland', 'player', 'wales', 'italy', 'try', 'win', 'play', 'injury', 'team', 'lion', 'cup', 'williams', 'wilkinson', 'squad', 'year', 'centre', 'jones', 'captain', 'penalty', 'kick', 'new', 'match', 'irish', 'week', 'leicester', 'bath', 'good', 'come', 'saturday', 'time', 'hodgson', 'zealand', 'think', 'fly', 'andy', 've', 'international', 'driscoll', 'look', 'referee', 'henson', 'woodward', 'ruddock', 'victory', 'line', 'club', 'gara', 'miss', 'great', 'minute', 'lose', 'tell', 'add', 'thomas', 'cardiff', 'scrum', 'season', 'mike', 'welsh', 'want']
    
    
    Topic 3:
    ['kenteris', 'greek', 'thanou', 'iaaf', 'drug', 'test', 'charge', 'athens', 'dope', 'ban', 'sprinter', 'olympics', 'tribunal', 'athlete', 'federation', 'miss', 'decision', 'olympic', 'evidence', 'kostas', 'pair', 'athletic', 'suspend', 'sport', 'appeal', 'athletics', 'committee', 'conte', 'hearing', 'trial', 'face', 'balco', 'fail', 'case', 'body', 'avoid', 'anti', 'official', 'clear', 'crash', 'withdraw', 'year', 'august', 'court', 'sydney', 'duo', 'find', 'expect', 'statement', 'silver', 'rule', 'claim', 'authority', 'association', 'present', 'medal', 'collin', 'coach', 'date', 'december', 'decide', 'game', 'allegation', 'issue', 'steroid', 'gold', 'tell', 'international', 'hear', 'deny']
    
    
    


```python
nmf_topic_matrix = nmf.transform(X)
nmf_topic_matrix.shape
```




    (471, 4)




```python
nmf_topic_matrix[32]
```




    array([0.        , 0.        , 0.        , 0.49252081])




```python
# count for topic 0
count_0=0
ls_0=[]
# count for topic 1
count_1=0
ls_1=[]
# count for topic 2
count_2=0
ls_2=[]
# count for topic 3
count_3=0
ls_3=[]
for i in range(nmf_topic_matrix.shape[0]):
    if np.argmax(nmf_topic_matrix[i])==0:
        count_0+=1
        ls_0.append(i+1)
    elif np.argmax(nmf_topic_matrix[i])==1:
        count_1+=1
        ls_1.append(i+1)
    elif np.argmax(nmf_topic_matrix[i])==2:
        count_2+=1
        ls_2.append(i+1)
    else:
        count_3+=1
        ls_3.append(i+1)
print("Topic 0 (Tennis): " + str(count_0)+ "  " , "Topic 1 (Cricket): " + str(count_1)+"  ", "Topic 2(Football): " + str(count_2)+"  ", "Topic 3(athletics): " + str(count_3))
print()
print("Topic 0: ", ls_0)
print()
print("Topic 1: ", ls_1)
print()
print("Topic 2: ", ls_2)
print()
print("Topic 3: ", ls_3)
```

    Topic 0 (Tennis): 171   Topic 1 (Cricket): 121   Topic 2(Football): 150   Topic 3(athletics): 29
    
    Topic 0:  [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 34, 37, 38, 40, 41, 42, 43, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 83, 84, 85, 86, 87, 90, 93, 94, 95, 96, 97, 98, 99, 100, 101, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 394, 395, 396, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471]
    
    Topic 1:  [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]
    
    Topic 2:  [2, 178, 180, 197, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371]
    
    Topic 3:  [4, 16, 17, 23, 26, 33, 35, 36, 39, 44, 45, 46, 47, 48, 49, 50, 51, 54, 79, 80, 81, 82, 88, 89, 91, 92, 389, 397, 445]
    


```python
# using the LDA 
```


```python
from sklearn.decomposition import LatentDirichletAllocation
```


```python
lda_model = LatentDirichletAllocation(n_components =4, max_iter=500, random_state=20)
```


```python
lda_model.fit(X)
```




    LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                              evaluate_every=-1, learning_decay=0.7,
                              learning_method='batch', learning_offset=10.0,
                              max_doc_update_iter=100, max_iter=500,
                              mean_change_tol=0.001, n_components=4, n_jobs=None,
                              perp_tol=0.1, random_state=20, topic_word_prior=None,
                              total_samples=1000000.0, verbose=0)




```python
lda_model.components_.shape
```




    (4, 1000)




```python
terms = vectorizer.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    term_comp = zip(terms, comp)
    sorted_terms = sorted(term_comp, key = lambda x:x[1], reverse=True)[:70]
    
    print("Topic "+ str(i)+ ":")
    topics=[]
    for item in sorted_terms:
        topics.append(item[0])
    
    print(topics)
    print("\n")
```

    Topic 0:
    ['win', 'play', 'year', 'england', 'game', 'match', 'world', 'good', 'team', 'player', 'final', 'day', 'test', 'time', 'set', 'second', 'cricket', 'new', 'open', 'rugby', 'injury', 'come', 'coach', 'cup', 'run', 'tour', 'champion', 'international', 'think', 'start', 'series', 'south', 'beat', 'seed', 'week', 'season', 'victory', 'great', 'race', 'title', 'captain', 'lose', 'old', 'add', 'break', 'ball', 've', 'zealand', 'africa', 'look', 'want', 'indoor', 'australian', 'number', 'australia', 'man', 'end', 'olympic', 'know', 'championship', 'record', 'return', 'event', 'work', 'tell', 'feel', 'european', 'like', 'way', 'month']
    
    
    Topic 1:
    ['pakistan', 'india', 'wicket', 'mohammad', 'khan', 'xi', 'ponting', 'australia', 'shoaib', 'gillespie', 'ganguly', 'inzamam', 'warne', 'ul', 'hayden', 'langer', 'lee', 'gilchrist', 'clarke', 'wkt', 'capt', 'mcgrath', 'catch', 'shane', 'singh', 'bangladesh', 'kumble', 'akhtar', 'vettori', 'test', 'ricky', 'kaif', 'tendulkar', 'butt', 'martyn', 'younis', 'kaneria', 'ball', 'pathan', 'razzaq', 'youhana', 'sami', 'dravid', 'haq', 'paceman', 'katich', 'glenn', 'sehwag', 'brett', 'hameed', 'lehmann', 'spinner', 'salman', 'abdul', 'kasprowicz', 'yasir', 'yousuf', 'danish', 'lbw', 'stump', 'bowl', 'matthew', 'bat', 'adam', 'run', 'michael', 'innings', 'inning', 'damien', 'boundary']
    
    
    Topic 2:
    ['drug', 'kenteris', 'iaaf', 'greek', 'thanou', 'dope', 'ban', 'conte', 'charge', 'balco', 'test', 'sprinter', 'olympics', 'athens', 'federation', 'evidence', 'athlete', 'tribunal', 'collin', 'suspend', 'committee', 'hearing', 'anti', 'case', 'decision', 'steroid', 'miss', 'appeal', 'trial', 'allegation', 'kostas', 'athletic', 'pair', 'olympic', 'body', 'athletics', 'sport', 'statement', 'fail', 'avoid', 'date', 'collins', 'court', 'white', 'official', 'face', 'clear', 'crash', 'sydney', 'withdraw', 'august', 'duo', 'authority', 'association', 'find', 'expect', 'silver', 'television', 'present', 'rule', 'hear', 'san', 'claim', 'refuse', 'december', 'issue', 'spend', 'light', 'deny', 'use']
    
    
    Topic 3:
    ['ireland', 'wale', 'robinson', 'half', 'france', 'nation', 'england', 'wales', 'italy', 'penalty', 'scotland', 'bath', 'wilkinson', 'centre', 'wasp', 'kick', 'try', 'ruddock', 'gara', 'leicester', 'driscoll', 'scrum', 'stade', 'flanker', 'henson', 'gloucester', 'hodgson', 'newcastle', 'wing', 'laporte', 'fly', 'sale', 'welsh', 'prop', 'cardiff', 'thomas', 'murphy', 'tait', 'lewsey', 'referee', 'biarritz', 'jonny', 'tindall', 'barkley', 'mike', 'replacement', 'cueto', 'goal', 'dawson', 'lock', 'kaplan', 'irish', 'ulster', 'line', 'pack', 'charlie', 'minute', 'jones', 'bortolami', 'williams', 'twickenham', 'corry', 'white', 'capt', 'davy', 'saturday', 'easterby', 'brian', 'leed', 'rome']
    
    
    


```python
lda_topic_matrix = lda_model.transform(X)
lda_topic_matrix.shape
```




    (471, 4)




```python
lda_topic_matrix[32]
```




    array([0.16113448, 0.03626693, 0.76635699, 0.03624159])




```python
# count for topic 0
count_0=0
ls_0=[]
# count for topic 1
count_1=0
ls_1=[]
# count for topic 2
count_2=0
ls_2=[]
# count for topic 3
count_3=0
ls_3=[]
for i in range(lda_topic_matrix.shape[0]):
    if np.argmax(lda_topic_matrix[i])==0:
        count_0+=1
        ls_0.append(i+1)
    elif np.argmax(lda_topic_matrix[i])==1:
        count_1+=1
        ls_1.append(i+1)
    elif np.argmax(lda_topic_matrix[i])==2:
        count_2+=1
        ls_2.append(i+1)
    else:
        count_3+=1
        ls_3.append(i+1)
print("Topic 0 : " + str(count_0)+ "  " , "Topic 1 (Cricket): " + str(count_1)+"  ", "Topic 2(Athletics): " + str(count_2)+"  ", "Topic 3(Football): " + str(count_3))
print()
print("Topic 0: ", ls_0)
print()
print("Topic 1: ", ls_1)
print()
print("Topic 2: ", ls_2)
print()
print("Topic 3: ", ls_3)
```

    Topic 0 : 408   Topic 1 (Cricket): 17   Topic 2(Athletics): 21   Topic 3(Football): 25
    
    Topic 0:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 83, 84, 85, 86, 87, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 147, 148, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 192, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 264, 265, 267, 269, 270, 271, 272, 273, 275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294, 295, 297, 298, 301, 302, 303, 304, 306, 307, 308, 309, 310, 311, 313, 314, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 336, 337, 338, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471]
    
    Topic 1:  [104, 108, 110, 121, 122, 123, 124, 139, 146, 149, 153, 189, 191, 193, 205, 208, 219]
    
    Topic 2:  [16, 17, 23, 26, 33, 35, 44, 45, 47, 48, 49, 50, 51, 54, 79, 80, 81, 82, 88, 89, 91]
    
    Topic 3:  [232, 242, 245, 254, 260, 261, 262, 263, 266, 268, 274, 279, 293, 296, 299, 300, 305, 312, 315, 323, 332, 333, 334, 335, 341]
    


```python

```
