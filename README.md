# Categorization-of-Sport-Articles-


## Problem Statement

This project aims to classify various bbc sports articles into different categories. I have used three different approaches to this topic modelling poblem; LSA (Latent Samentic Analysis), NMF (Non Negative Factorization) and LDA (Latent Dirichlet Allocation) methods. I have used nltk, spcay, numpy, regex, pandas, matplotlib, os and sns library to assist me with various operations. It is an unsupervised problem.

## Dataset

The dataset contains 471 articles related to different sports. They are in the .txt format and named as 001.txt, 002.txt and so on. 

## Preprocessing and cleaning 

Here I have removed all unnecessary information which dosen't contribute to classify these articles. I have removed new line characters, numbers, punctuations, special characters, single alphabets, stop words and "-PRON-" string which appears a lot by seeing through the frequency distribution table of the list of words and performed lemmatization as well. 

Frequency Distribution graph of words present in document corpus (for top 30 words)

Before Tokenization and stop words removal 

![1](https://user-images.githubusercontent.com/36281158/86002547-53921580-b9c5-11ea-87d5-f92c44d340da.PNG)

After doing lemmatization and stop words removal

![2](https://user-images.githubusercontent.com/36281158/86002745-a10e8280-b9c5-11ea-8076-b263d1b18865.PNG)

After removing "-PRON-" removal as it dosen't convey anything

![3](https://user-images.githubusercontent.com/36281158/86002866-d0bd8a80-b9c5-11ea-996c-11e9ddadf1fb.PNG)

Now it is ready for Topic Modelling 

## Topic Modelling 

Here document term matrix is created first. Document term matrix is an array where rows are the documents and columns are the words used in these document matrix. Since the matrix will be largely sparse,  keep only important words that will help in classifying these articles. I have used Tfidf vectorizer method to give values in the given matrix.
Tf-Idf stands for term frequency inverse document frequency. Term Frequency summarizes how often a given word appears within a document and Inverse Document Frequency downscales words that appear a lot across documents. Tfidf scores are the product of these two terms. Without inverse document term, words which appear more common (that carries little information) would be weighed higher and words appearing rare(which contains more information) would be weighed less. 

Inverse document frequency is defined as:

idf(t,D)=log (N/ |d∈D:t∈d|) 

Where  N  is the total number of documents in the corpus, and  |d∈D:t∈d|  is the number of documents where  t  appears

A vecotizer object is created by calling the TfidfVectorizer class. Some arguments for it are as follows.

max_features = 1000, it meanns we want only 1000 words to help us in classifying the articles 

min_df = 5, it means we are keeping those words which have appeared more than or equal to 5 times in the document corpus 

max_df= 0.9, it means removing those words which have appeared in more than 90 % of the documents. 

### LSA (Latent Samentic Analysis) Approach 

The next step is to represent each and every term and document as a vector.

Specify the number of topics (k=4)

Decompose the document-term matrix into 2 matrix.

Document-Topic Matrix
Topic-Term Matrix


![4](https://user-images.githubusercontent.com/36281158/86008827-30b82f00-b9ce-11ea-9d67-3ab9058ecd87.PNG)

A is the document term matrix which is decomposed to U(document - topic matrix), S and V.T(Topic-term matrix). Here the truncated single value decompistion method is used for decomposing the matrix. 

This method of decomposition gives us the unique U, S and V for A. The rows and columns of U are orthogonal, S is a diagonal matrix (with values in the decreasing order which signifies the relative importance of topics in the document corpus) and V ( which shows waords carrying importance in correspondoing topics)

Applying the vectorizer over the "cleaned" dataset and then applying the LSA model gives us the following results.

Topics numbered 0-4 are shown below with their corresponding top 70 words describing them. 

![5](https://user-images.githubusercontent.com/36281158/86010326-1bdc9b00-b9d0-11ea-8239-24939740190e.PNG)

We can observe topic 0 contains words which decribe cricket, topic 1 has words which describes tennis, topic 2 has words which describe football and topic 3 has words which describes athletics. 

We can get the values of a single document with its topics. 

lsa_topic_matrix[32]      array([ 0.16812655,  0.10387154, -0.30388207,  0.48508372])

We can see that the document 32 has values given in the array. We see that the fourth value (0.48) is highest so we can say that document 32 belongs to Topic 3. Doing this for all the documents we get

Topic 0 (Cricket): 425   Topic 1 (Tennis): 17   Topic 2(Football): 3   Topic 3(athletics): 26

## Non Negative Matrix Factorization Approach

We see that one value in the above array was negative. So this method gives us the two matrices which have non negative values. 

Rather than constraining our factors to be orthogonal, another idea would to constrain them to be non-negative. NMF is a factorization of a non-negative data set 𝑉:
𝑉=𝑊𝐻
into non-negative matrices (W,H)

### Intution 


![6](https://user-images.githubusercontent.com/36281158/86011607-c0131180-b9d1-11ea-9870-48514c9b6342.PNG)

Nonnegative matrix factorization (NMF) is a non-exact factorization that factors into one skinny positive matrix and one short positive matrix. NMF is NP-hard and non-unique

![7](https://user-images.githubusercontent.com/36281158/86011715-eb95fc00-b9d1-11ea-9156-071a990ac789.PNG)

Applying the vectorizer over the "cleaned" dataset and then applying the NMF model gives us the following results.

Topics numbered 0-4 are shown below with their corresponding top 70 words describing them

![8](https://user-images.githubusercontent.com/36281158/86012541-f3a26b80-b9d2-11ea-9276-648c98859756.PNG)

We can observe topic 0 contains words which decribe tennis, topic 1 has words which describes cricket, topic 2 has words which describe football and topic 3 has words which describes athletics 

We can get the values of a single document with its topics. 

nmf_topic_matrix[32]      array([ 0.        , 0.        , 0.        , 0.49252081])

We can see that the document 32 has values given in the array. We see that the fourth value (0.49) is highest so we can say that document 32 belongs to Topic 3. Also we can see that no negative value is present.  Doing this for all the documents we get

Topic 0 (Tennis): 171   Topic 1 (Cricket): 121   Topic 2(Football): 150   Topic 3(athletics): 29

Also comparing the Cricket topic words list for NMF and LSA, we can the words are little bit different and their importance is also changed(by seeing the relative positons in the array). Also we can see that NMF has given most importance to topic 0 which has most of the words belonging to tennis whereas LSA has given most importance to topic 0 which has most of the words belonging to cricket. 

## Latent Dirichelt Allocation Approach 

It is a generative probabilistic model. It assumes documents are a mixture of topics and topics are the mixtures of words. Documents are probabilistic distribution of topics and topics are probabilistic distribution of words. Goal of this method is to find optimized representations of these. 

![9](https://user-images.githubusercontent.com/36281158/86016470-a70d5f00-b9d7-11ea-9f55-10528cf4f6e7.PNG)

In the diagram every word in a document is related to the hidden topic Z. Theta represents the topic word distribution in the corpus. Alpha and beta are hyperparameters of the mode. Alpha controls per document topic distribution and beta controls per topic word distribution. 

![10](https://user-images.githubusercontent.com/36281158/86017760-4848e500-b9d9-11ea-88ac-99e8d4a10351.PNG)


K1, k2 .... are the random topics assigned, D1, D2... are the documents and W1, W2.... are the words in them

Now the optimization step. Assuming for the current word, topic assigned is not correct whereas for every other words topics assigned are correct. 

for each document -> for each word : a probability p is calculated.

p = p1 * p2

![11](https://user-images.githubusercontent.com/36281158/86018107-bb525b80-b9d9-11ea-9b36-e3fb256ab8ad.PNG)

Now using p1 and p2, a new product is obtained which assigns the word a new topic k'.  Now iteratiosn are being done till the steady state is achieved. 

Applying the vectorizer over the "cleaned" dataset and then applying the LDA model gives us the following results.

Topics numbered 0-4 are shown below with their corresponding top 70 words describing them
    
![12](https://user-images.githubusercontent.com/36281158/86019073-002ac200-b9db-11ea-83ab-a3fe72b7315e.PNG)

We can observe topic 0 contains words which decribe nothing, topic 1 has words which describes cricket, topic 2 has words which describe athletics and topic 3 has words which describes football 

We can get the values of a single document with its topics.

lda_topic_matrix[32]      array([0.16113448, 0.03626693, 0.76635699, 0.03624159])

We can see that the document 32 has values given in the array. We see that the third value (0.76) is highest so we can say that document 32 belongs to Topic 2.Doing this for all the documents we get

Topic 0 : 408   Topic 1 (Cricket): 17   Topic 2(Athletics): 21   Topic 3(Football): 25

We can see that this model has topics which are not clearly defined by the words descibing them specially in the case for topic 0 as compared to NMF and LSA

#### So in comparison, we see that LSA performs better than the two as it helps us in distinguishing the articles more by seeing throught the words in topics

PS: These results are different for different datasets. My motive of doing this project is use three techinques used in this process and learn them.


































