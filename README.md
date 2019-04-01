# as-you-like-it
Step 1. Fetch the data from [ratings](https://github.com/zygmuntz/goodbooks-10k/blob/master/ratings.csv).
The ratings data look like this:

user_id,book_id,rating
1,258,5
2,4081,4
2,260,5
2,9296,5
2,2318,3

Ratings go from one to five. Both book IDs and user IDs are contiguous. For books, they are 1-10000, for users, 1-53424.

[to_read.csv](https://github.com/zygmuntz/goodbooks-10k/blob/master/to_read.csv) provides IDs of the books marked "to read" by each user, as user_id,book_id pairs, sorted by time. There are close to a million pairs.

[books.csv](https://github.com/zygmuntz/goodbooks-10k/blob/master/books.csv) has metadata for each book (goodreads IDs, authors, title, average rating, etc.). The metadata have been extracted from goodreads XML files, available in books_xml.
