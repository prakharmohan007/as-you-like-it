

# as-you-like-it
Step 1. Fetch the data from [ratings](https://github.com/zygmuntz/goodbooks-10k/blob/master/ratings.csv).
The ratings data look like this:

user\_id,book\_id,rating

1,258,5

2,4081,4


Ratings go from one to five. Both book IDs and user IDs are contiguous. For books, they are 1-10000, for users, 1-53424.

[to\_read.csv](https://github.com/zygmuntz/goodbooks-10k/blob/master/to_read.csv) provides IDs of the books marked "to read" by each user, as user_id,book_id pairs, sorted by time. There are close to a million pairs.

[books.csv](https://github.com/zygmuntz/goodbooks-10k/blob/master/books.csv) has metadata for each book (goodreads IDs, authors, title, average rating, etc.). The metadata have been extracted from goodreads XML files, available in books_xml.

Step2: We have modified the ratings to give new\_ratings.csv which contains 10k users and 6176 books. For these books the summary is stored in the file new\_books.csv and the embedding is stoed in file summary\_embeddings.txt (use pickle to load it). Note that the length of embeddings is 125. 

Step3: We have ran CF models and got RMSE (Cosine) =0.8833 and RMSE(Pearson)=0.8708

Step4: 
