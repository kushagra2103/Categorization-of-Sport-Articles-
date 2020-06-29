# Categorization-of-Sport-Articles-


## Problem Statement

This project aims to classify various bbc sports articles into different categories. I have used three different approaches to this topic modelling poblem; LSA (Latent Samentic Analysis), NMF (Non Negative Factorization) and LDA (Latent Dirichlet Allocation) methods. I have used nltk, spcay, numpy, regex, pandas, matplotlib, os and sns library to assist me with various operations. It is an unsupervised problem.

## Dataset

The dataset contains 471 articles related to different sports. They are in the .txt format and named as 001.txt, 002.txt and so on. 

## Preprocessing and cleaning 

Here I have removed all unnecessary information which dosen't contribute to classify these articles. I have removed new line characters, numbers, punctuations, special characters, single alphabets, stop words and "-PRON-" string which appears a lot by seeing through the frequency distribution table of the list of words and performed lemmatization as well. 

