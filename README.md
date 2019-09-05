<center> <h1>As You Like It: Book Recommender System</h1> </center>

# Overview
Most of the book recommender systems use Collaborative Filtering or Matrix Factorization to model the user's preference to the books. However one important feature that is typically ignored by recommender systems is the summary of the book itself. In this project, we propose a personalized book recommender system based on Bayesian Personalized Ranking (BPR) that uses summaries of books as one of the key features. We also, propose a novel way to modify user-user collaborative filtering using the summary embeddings as an additional weight of item-item similarity and comparing the results with the traditional collaborative filtering.

# Summary as a Feature
The summary of a book gives a better understanding of the whole content of the book and would thus be more informative in finding similar books according to user’s taste. We have used the embeddings generated from summaries as feature in User-Item combined CF and S-BPR. For User-Item combined CF, the embeddings are used to find item item similarity. Item-Item similarity is calculated as cosine similarity between 125 dimensional embedding vectors of the two books . These similarities are then used as explained under user-item combined collaborative filtering.

# User-Item Combined Collaborative Filtering
We introduce an item-item similarity term in user-user collaborative filtering as depicted in the diagram below.

![Modified CF](https://drive.google.com/uc?export=view&id=1DmQUYqy8dPd-5Z4VAyNW0tdyjLfKwvsr)

The motivation and the equation can be understood by the following example. Consider that the rating of user 'i' is to be predicted for item "Harry Potter and the chamber of secrets". The user 'j' has a similarity of 0.9 with user 'i' but has rated "Harry potter and the philosopher's stone" which is 0.9 similar to the former. Simple User-User CF would have ignored this user for not rating the required book even though his rating can be used for prediction. Our proposed modification will use both the similarities and will weight the rating of user 'j' for the later book by a factor of 0.9x0.9 = 0.81.

# Experiment
In this project, we have used collaborative filtering and Bayesian Personalized Ranking to evaluate the importance of book summaries for book recommendation systems.

## Dataset
The dataset used is [10K Goodbooks (Goodreads)](http://fastml.com/goodbooks-10k-a-new-dataset-for-book-recommendations/) which consists of 10,000 books with 6 million ratings by over 53,000 unique users. The dataset contains metadata of books like author, year of publication, genre, goodreads ID, etc. The explicit ratings include range from 1 to 5. The dataset was published on Kaggle in 2017. For the purpose of this project, we have considered ratings of only 10,000 users. The training set has 976,982 samples and test set has 325,661 samples.

## Summary
The [wikipedia-api](https://pypi.org/project/wikipedia/) in python is used to find summaries/plots of the books of the dataset. This library allows searching on Wikipedia and parsing of data such as summaries, links, images, etc. A single string including the title of the book, along with the author and year of publication was used for the search. If no Wikipedia page exists for a book then it is ignored in our project. This reduces the number of books from 10,000 to 6,176 books.

## Summary Embeddings
[Gensim](https://radimrehurek.com/gensim/) is a production-ready open-source library for unsupervised topic modeling and natural language processing, using modern statistical machine learning. We have used [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) model from gensim to generate embeddings for books using the summaries captured through wikipedia-api. Each summary is represented by a vector embedding of length 125 and these vector representations are then used to find the similarities in the content of books.

## Execution
The file [cf_new.py](https://github.com/prakharmohan007/as-you-like-it/blob/master/cf_new.py) incorporates the training, prediction and evaluation of modified collaborative filtering.

### Train
The CombinedCollaborativeFiltering::fit(...) function calculates the user-user and item-item similarity. It also provides options on the similarity criteria (Cosine similarity, Pearson's Correlation similarity) and is flexible to add more options in the future.

The function also provides an option to save the calculated similarity matrices to save time and computation when training is not necessary.
### Test
The CombinedCollaborativeFiltering::predict(...) function predicts the rating for user-item pairs in the test set. The similarity matrices can either be trained from scratch or an existing model can be loaded using CombinedCollaborativeFiltering::load_similarity_matrix(...).

The CombinedCollaborativeFiltering::evaluate(...) calculates the RMSE and MAE values on the test set.

## Future Possibilities
We believe that the performance of our methods can be significantly improved through many ways. The hyper-parameter $K$ in modified CF can be replaced by weighted values of similarity scores. Moreover, for many books we could not generate summaries due to the lack of wikipedia pages or due to multiple pages-disambiguation error. We believe that usage of accurate summaries and new embeddings generation methods such as BERT can further improve the results.

## Resources
The source code of this project can be accessed from [github](https://github.com/prakharmohan007/as-you-like-it). For detailed reference, have a look at the [project report](https://drive.google.com/file/d/1PJ0ahwNWjEVqoxVPqT9KMWaLqyXrppP7/view?usp=sharing)
