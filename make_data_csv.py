# This file extracts the first 'U' users and first 'B' books
# and generate a new csv file with a subset of original data

import csv
import os
import argparse

def create_subdata(num_user = 2000, num_item = 2000, path = "./data"):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError as er:
        print ("Error with directory ", path, ". Error message: ", err)
        path = "./"
    
    with open('./goodbooks-10k/ratings.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []  # list(csv_reader)
        head = 0
        for row in csv_reader:
            # print(data)
            if head == 0:
                head = 1
                continue
    
            if int(row[0]) <= num_users and int(row[1]) <= num_items:
                data.append(row)
                # print(row)
    
    filepath = path+"/ratings_"+str(num_users)+"_"+str(num_items)+".csv"
    with open(filepath, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in data:
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
