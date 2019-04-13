# This file extracts the first 'U' users and first 'B' books
# and generate a new csv file with a subset of original data

import csv
import os
import argparse

def get_rating_freq(data, column):
    freq = {}
    # np_data = np.array(user_freq)
    # num_unique_users = np.unique(np_data[:,0]).shape[0]

    for row in data:
        if row[column] in freq:
            freq[row[column]] += 1
        else:
            freq[row[column]] = 1
    
    # top_keys = sorted(freq, key = freq.get, reverse=True)
    # print(top_keys)
    # return top_keys
    return freq

def create_subdata(num_user = 2000, num_item = 2000, path = "./data"):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError as er:
        print ("Error with directory ", path, ". Error message: ", err)
        path = "./"
   
    book_rating_th = 4
    
    # READ DATA
    data = []
    with open('./goodbooks-10k/ratings.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)
    
    header = data[0]
    del data[0]
    print("CHECK: number of samples: ", len(data))
    data = [list( map(int,i) ) for i in data]

    # Get users with most number of ratings
    freq_users = get_rating_freq(data,0)
    sorted_user_freq = sorted(freq_users, key = freq_users.get, reverse=True)
    # select top num_users number of users
    top_users_list = sorted_user_freq[:num_users]
    top_users = set(top_users_list)
    print("CHECK: number of top users: ",len(top_users))
    # print(top_users)
 
    # adjust user-ids to 0:num_users
    # uid_map = {}
    # uid = 0
    # for u in top_users_list:
    #     uid_map[u] = uid
    #     uid += 1

    # filter the data based to top users
    intermediate_data = []
    for row in data:
        if row[0] in top_users:
            # print(row)
            intermediate_data.append(row)
    print("CHECK: Number of samples after filtering: ", len(intermediate_data))
    
    # get books with most number of ratings by above filtered users
    freq_book = get_rating_freq(intermediate_data,1)
    atleast_two_rating = dict(filter(lambda x: (x[1]) > book_rating_th, freq_book.items()))
    sorted_book_freq = sorted(atleast_two_rating, key = atleast_two_rating.get, reverse=True)
    # books = list(atleast_two_books.keys())
    # select top num_items number of items
    top_books_list = sorted_book_freq[:num_items]
    top_books = set(top_books_list)
    print("CHECK: number of top rated books: ", len(top_books))
    # print(top_books)

    # adjust book ids to 0:num_items
    # bid_map = {}
    # bid = 0
    # for b in top_books_list:
    #     bid_map[b] = bid
    #     bid += 1

    # filter data again
    sub_data = []
    for row in intermediate_data:
        if row[1] in top_books:
            # print(row)
            sub_data.append(row)
    print("CHECK: number of samples: ", len(sub_data))

    numi = min(num_items, len(top_books))
    numu = min(num_users, len(top_users))

    # save as csv file
    # sub_data = [list( map(str,i) ) for i in sub_data]
    filepath = path+"/ratings_"+str(numu)+"_"+str(numi)+"_"+str(book_rating_th+1)+".csv"
    with open(filepath, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        for row in sub_data:
            # user = uid_map[row[0]]
            # book = bid_map[row[1]]
            # row_ = [str(user), str(book), str(row[2])]
            csv_writer.writerow(row)
    
    # print("completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_user', help='number of users to extract', type=int, default=2000)
    parser.add_argument('--num_item', help='number of movies to items', type=int, default=2000)
    parser.add_argument('--dir', help='target directory for csv', default="./data")
    args = parser.parse_args()

    path = args.dir
    num_users = args.num_user
    num_items = args.num_item

    # print("path: ", path, ", num users: ", num_users, ", num items: ", num_items)
    create_subdata(num_users, num_items, path)
    print("completed!")
