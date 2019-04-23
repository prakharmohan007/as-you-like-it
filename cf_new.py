import numpy as np
import csv
import pandas as pd
import math
from sklearn.model_selection import train_test_split


class CombinedCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.ratings = np.zeros((num_users, num_items))  # num_users x num_items
        self.train_ratings = []
        self.test_ratings = []
        self.user_idx = {}
        self.item_idx = {}
        self.user_data = {}
        self.similarity_matrix = np.zeros((num_users, num_users))

    def load_rating_csv(self, rating_file):
        with open(rating_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)

        del data[0]
        print("[CCF] load_rating_csv: number of samples: ", len(data))

        # convert to int
        data = [list(map(int, i)) for i in data]

        i_idx = 0
        u_idx = 0
        temp_i = 0
        temp_u = 0
        for sample in data:
            if sample[0] in self.user_idx:
                temp_u = self.user_idx[sample[0]]
            else:
                self.user_idx[sample[0]] = u_idx
                self.user_data[u_idx] = {"num_ratings": 0, "sum_rating": 0, "sum_rating_sq": 0}
                temp_u = u_idx
                u_idx += 1

            if sample[1] in self.item_idx:
                temp_i = self.item_idx[sample[1]]
            else:
                self.item_idx[sample[1]] = i_idx
                temp_i = i_idx
                i_idx += 1

            self.ratings[temp_u, temp_i] = sample[2]
            self.user_data[temp_u]["num_ratings"] += 1
            self.user_data[temp_u]["sum_rating"] += sample[2]
            self.user_data[temp_u]["sum_rating_sq"] += sample[2]*sample[2]

        print("[CCF] load_rating_csv: number of users: ", u_idx)
        print("[CCF] load_rating_csv: number of items: ", i_idx)
        print("[CCF] load_rating_csv: rating matrix made")

        # data = pd.read_csv(rating_file)
        # print("[CCF] load_rating_csv: number of samples: ", len(data))

        # self.ratings = data.groupby(['user_id', 'book_id']).rating.mean().unstack(fill_value=-1)
        # print("[CCF] load_rating_csv: Data has", self.ratings.shape[0], "Rows,", self.ratings.shape[1], "Columns")

        # print("[CCF] load_rating_csv: data frame")
        # print(self.ratings[:])

        # train_data, test_data = train_test_split(data, train_size=0.9, random_state=20)
        # self.train_ratings = train_data.groupby(['user_id', 'book_id']).rating.mean().unstack(fill_value=-1)
        # self.test_ratings = test_data.groupby(['user_id', 'book_id']).rating.mean().unstack(fill_value=-1)
        # print("[CCF] load_rating_csv: Training matrix shape: ", self.train_ratings.shape[0], "Rows,",
        #       self.train_ratings.shape[1], "Columns")
        # print("[CCF] load_rating_csv: Testing matrix shape", self.test_ratings.shape[0], "Rows,",
        #       self.test_ratings.shape[1], "Columns")
        #
        # print("[CCF] load_rating_csv: prepare training sample's list")

    # find cosine similarity for user i and j
    # input: user id i and j
    def calc_cosine_similarity(self, i, j):
        uid_i = self.user_idx[i]
        uid_j = self.user_idx[j]

        rating_i = np.array(self.ratings[i, :], copy=True)
        rating_j = np.array(self.ratings[j, :], copy=True)

        # np.where(rating_i==-1, 0, rating_i)
        rating_i[rating_i == -1] = 0
        # np.where(rating_j==-1, 0, rating_j)
        rating_j[rating_j == -1] = 0

        dot_prod = np.dot(rating_i, rating_j)
        similarity = dot_prod / (math.sqrt(self.user_data[i]["sum_rating_sq"]) *
                                 math.sqrt(self.user_data[j]["sum_rating_sq"]))
        return similarity

    def get_cosine_similarity_matrix(self):
        # print(simi_matrix.shape)
        for i in range(self.num_users):
            for j in range(i + 1, self.num_users):
                sim = self.calc_cosine_similarity(i, j)
                self.similarity_matrix[i, j] = sim
                self.similarity_matrix[j, i] = sim
        # print(simi_matrix[530:, 530:])

    # method: method used to calculate user-user similarity. 1-> cosine similarity, 2->pearson similairity
    def fit(self, method=1):
        pass

    def predict_rating(self, user_i, item_a):
        k = 0.001

        for j in range(self.num_users):
            if j != user_i and self.ratings[j, item_a] != 0:
                item_b = item_a
            elif j != user_i and self.ratings[j, item_a] == 0:
                # find most similar item 

    def predict_rating(self, user_i, movie_j):
        # calc similarity*rating diff for all users
        k = 0.05
        sum_term = 0
        for i in range(self.num_users):
            if i != user_i and self.ratings[i, movie_j] != -1:
                # if i != user_i:
                sum_term += simi_matrix[user_i, i] * (rating_data[i, movie_j] - user_data[i]["avg_rating"])

        rating = user_data[user_i]["avg_rating"] + k * sum_term
        return rating


if __name__ == "__main__":
    obj_ccf = CombinedCollaborativeFiltering(10000, 6176)
    obj_ccf.load_rating_csv("new_ratings.csv")
    # print(obj_ccf.ratings[2, :])
