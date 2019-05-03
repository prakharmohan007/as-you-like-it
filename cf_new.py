import numpy as np
from numpy import linalg as LA
import csv
import pandas as pd
import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from gensim.test.utils import get_tmpfile
import gensim


class CombinedCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        # self.ratings = np.zeros((num_users, num_items))  # num_users x num_items
        self.train_ratings = np.zeros((num_users, num_items))  # rating matrix (num_users x num_items)
        self.test_data = []  # list of samples
        self.user_idx = {}  # { user_ids : index }
        self.item_idx = {}  # { item_ids : index }
        self.user_data = {}  # {user_index: {"num_ratings":0, "sum_ratings":0 , "sum_sq_ratings":0 }}

        # similarities
        self.user_similarity_matrix = np.zeros((num_users, num_users))
        self.most_similar_users = np.zeros((num_users, num_users)).astype(int)  # stores index of most similar users
        self.item_similarity_matrix = np.zeros((num_items, num_items))
        self.most_similar_items = np.zeros((num_items, num_items)).astype(int)  # stores item_id of most similar item

        # embeddings files
        self.embed_file = ""
        self.embed_model_file = ""
        self.books_summary_file = ""

    def read_training_csv(self, rating_file):
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
                self.user_data[u_idx] = {"num_ratings": 0, "sum_ratings": 0, "sum_sq_ratings": 0}
                temp_u = u_idx
                u_idx += 1

            if sample[1] in self.item_idx:
                temp_i = self.item_idx[sample[1]]
            else:
                self.item_idx[sample[1]] = i_idx
                temp_i = i_idx
                i_idx += 1

            self.train_ratings[temp_u, temp_i] = sample[2]
            self.user_data[temp_u]["num_ratings"] += 1
            self.user_data[temp_u]["sum_ratings"] += sample[2]
            self.user_data[temp_u]["sum_sq_ratings"] += sample[2] * sample[2]

        print("[CCF] load_rating_csv: number of users: ", u_idx)
        print("[CCF] load_rating_csv: number of items: ", i_idx)
        self.num_users = u_idx
        self.num_items = i_idx
        print("[CCF] load_rating_csv: rating matrix made")

    def read_testing_csv(self, test_file):
        with open(test_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.test_data = list(csv_reader)

        del self.test_data[0]
        print("[CCF] read_testing_csv: number of test samples: ", len(self.test_data))

        # convert to int
        self.test_data = [list(map(int, i)) for i in self.test_data]

    def data_loader(self, train_file=None, test_file=None, embed_file=None, embed_model_file=None,
                    books_summary_file=None):
        if train_file is None:
            print("[CCF] data_loader: No csv for training provided")
            exit(-1)
        else:
            self.read_training_csv(train_file)

        if test_file is not None:
            self.read_testing_csv(test_file)
        else:
            print("[CCF] data_loader: No csv for test data provided")

        self.embed_file = embed_file
        self.embed_model_file = embed_model_file
        self.books_summary_file = books_summary_file

    # find cosine similarity for user i and j
    # input: user id i and j
    def calc_cosine_similarity(self, i, j):
        # uid_i = self.user_idx[i]
        # uid_j = self.user_idx[j]

        rating_i = np.array(self.train_ratings[i, :], copy=True)
        rating_j = np.array(self.train_ratings[j, :], copy=True)

        dot_prod = np.dot(rating_i, rating_j)
        similarity = dot_prod / (math.sqrt(self.user_data[i]["sum_sq_ratings"]) *
                                 math.sqrt(self.user_data[j]["sum_sq_ratings"]))
        return similarity

    # TODO
    # find pearson similarity for user i and j
    # input: user in
    def calc_pearson_similarity(self, i, j):
        return i * j

    # method: 1 -> cosine similarity, 2 -> pearson similarity
    def get_user_similarity_matrix(self, method=1):
        for i in range(self.num_users):
            for j in range(i, self.num_users):
                sim = 0
                if method == 1:
                    # calc cosine similarity
                    sim = self.calc_cosine_similarity(i, j)
                elif method == 2:
                    # calculate pearson similarity
                    sim = self.calc_pearson_similarity(i, j)

                self.user_similarity_matrix[i, j] = sim
                self.user_similarity_matrix[j, i] = sim
        print("[CCF] get_user_similarity_matrix: user-user similarity matrix made")

    # for every user, gives a list of similar users with most similar first and least similar in the last
    def get_most_similar_users(self):
        self.most_similar_users = np.fliplr(np.argsort(self.user_similarity_matrix))
    
    def read_corpus(self):
        x = pd.read_csv(self.books_summary_file)
        for index, row in x.iterrows():
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[1]), [row[0]])

    def calc_cosine_similarity_item(self, embedding_i, embedding_j):
        dot_prod = np.dot(embedding_i, embedding_j)
        similarity = dot_prod / ((LA.norm(embedding_i, ord=2)) * (LA.norm(embedding_j, ord=2)))
        return similarity

    def get_most_similar_items(self):
        self.most_similar_items = np.fliplr(np.argsort(self.item_similarity_matrix))

    # embed file -> embeddings file (contains embedding of all the books): summary_embeddings
    # embed_model_file -> Doc2Vec model based on the summaries: my_doc2vec_model
    # book_file -> books and summaries: new_books.csv
    # most_similar_item contains the indices and not the book_ids
    def get_item_similarity_matrix(self):
        # Load the embeddings from the file
        embeddings = np.loadtxt(self.embed_file)

        # Load model file
        # model = Doc2Vec.load(self.embed_model_file)
        # Load the new_books.csv which contains book id and summary

        # Author - Manish
        # document = []
        # book_index = []
        # summary_file = pd.read_csv(self.books_summary_file)
        # for index, row in summary_file.iterrows():
        #     line_list = row[1].split()
        #     document.append(line_list)
        #     book_index.append(self.item_idx[row[0]])
        #
        # tagged_data = [TaggedDocument(d, [i]) for d, i in zip(document, book_index)]
        # model = Doc2Vec(tagged_data, vector_size=125, workers=6)

        # fname = get_tmpfile("my_doc2vec_model")
        # model.save(fname)
        # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        # for row in range(self.num_items):
        #     new_vector = model.infer_vector(document[row])
        #     sims = model.docvecs.most_similar([new_vector], topn=self.num_items)
        #     for col in range(self.num_items):
        #         # sims[col][0] -> index of most similar item
        #         # sims[col][1] -> similarity
        #         self.most_similar_items[row][col] = sims[col][0]
        #         self.item_similarity_matrix[row][sims[col][0]] = sims[col][1]

        # Author: Vansh
        print("[CCF] get_item_similarity_matrix: creating corpus.....")
        corpus = list(self.read_corpus())
        print("[CCF] get_item_similarity_matrix: corpus created")

        # train gensim doc2vec
        print("[CCF] get_item_similarity_matrix: Training Gensim Doc2Vec....")
        model = gensim.models.doc2vec.Doc2Vec(corpus, vector_size=125, workers=6)
        print("[CCF] get_item_similarity_matrix: Gensim Doc2Vec trained....")

        print("[CCF] get_item_similarity_matrix: Get embedding vectors for all summaries")
        id_to_embeddings = {}  # book id: embedding
        for index in range(len(corpus)):
            id_to_embeddings[int(corpus[index].tags[0])] = model.infer_vector(corpus[index].words)
        print("[CCF] get_item_similarity_matrix: Embeddings generated, preparing similarity matrix")
        for id1 in id_to_embeddings:
            for id2 in id_to_embeddings:
                index1 = self.item_idx[id1]
                index2 = self.item_idx[id2]
                sim = self.calc_cosine_similarity_item(id_to_embeddings[id1], id_to_embeddings[id2])
                self.item_similarity_matrix[index1, index2] = sim
                self.item_similarity_matrix[index2, index1] = sim

        print("[CCF] get_user_similarity_matrix: user-user similarity matrix made")

    # method: method used to calculate user-user similarity. 1-> cosine similarity, 2->pearson similairity
    # matrix: 1 -> user_similarity, 2 -> item_similarity, 3 -> both
    def fit(self, method=1, matrix=3, save_file=None):
        self.user_similarity_matrix = np.ones((self.num_users, self.num_users))
        self.most_similar_users = np.zeros((self.num_users, self.num_users)).astype(int)
        # calculate user - user similaruty matrix
        if matrix == 1 or matrix == 3:
            print("[CCF] fit: preparing user-user similarity matrix....")
            self.get_user_similarity_matrix(method)
            print("[CCF] fit: user-user similarity matrix generated")
            print("[CCF] fit: preparing most similar user matrix....")
            self.get_most_similar_users()
            print("[CCF] fit: most similar user matrix generated")

            if save_file:
                print("[CCF] fit: writing user similarity matrix as npy....")
                np.save("user_similarity_matrix.npy", self.user_similarity_matrix)
                np.save("most_similar_users.npy", self.most_similar_users)
                print("[CCF] fit: writing user similarity matrix completed!")

        self.item_similarity_matrix = np.ones((self.num_items, self.num_items))
        self.most_similar_items = np.zeros((self.num_items, self.num_items)).astype(
            int)  # stores item_id of most similar item
        # calculate item - item similarity matrix
        if matrix == 2 or matrix == 3:
            print("[CCF] fit: preparing item - item similarity matrix using embeddings....")
            self.get_item_similarity_matrix()
            print("[CCF] fit: item - item similarity matrix generated")
            print("[CCF] fit: preparing most similar item matrix....")
            self.get_most_similar_items()
            print("[CCF] fit: most similar item matrix generated")

            if save_file:
                print("[CCF] fit: writing item similarity matrix as npy....")
                np.save("item_similarity_matrix.npy", self.item_similarity_matrix)
                np.save("most_similar_items.npy", self.most_similar_items)
                print("[CCF] fit: writing item similarity matrix completed!")

        print("[CCF] fit: Completed!")

    def load_similarity_matrices_csv(self, user_file, item_file):
        print("[CCF] load_similarity_matrices: loading user similarity matrix.....")
        with open(user_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)
        data = [list(map(float, i)) for i in data]
        self.user_similarity_matrix = np.array(data)
        print("[CCF] load_similarity_matrices: user similarity matrix loading complete!")

        print("[CCF] load_similarity_matrices: loading item similarity matrix.....")
        with open(item_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)
        data = [list(map(int, i)) for i in data]
        self.item_similarity_matrix = np.array(data)
        print("[CCF] load_similarity_matrices: item similarity matrix loading complete!")

    def load_similarity_matrix(self, user_file1="user_similarity_matrix.npy", user_file2="most_similar_users.npy",
                               item_file1="item_similarity_matrix.npy", item_file2="most_similar_items.npy"):
        # user similarity matrix
        print("[CCF] load_similarity_matrices: loading user similarity matrix.....")
        self.user_similarity_matrix = np.load(user_file1)
        print("[CCF] load_similarity_matrices: user similarity matrix loading complete!")

        if self.user_similarity_matrix.shape != (self.num_users, self.num_users):
            print("[CCF] load_similarity_matrix: user similarity matrix dimensions mismatch. Expected: ",
                  (self.num_users, self.num_users), ", Received: ", self.user_similarity_matrix.shape)
            exit(-1)

        # most similar user matrix
        print("[CCF] load_similarity_matrices: loading most similar user matrix.....")
        self.most_similar_users = np.load(user_file2)
        print("[CCF] load_similarity_matrices: most similar user matrix loading complete!")

        if self.most_similar_users.shape != (self.num_users, self.num_users):
            print("[CCF] load_similarity_matrix: most similar user matrix dimensions mismatch. Expected: ",
                  (self.num_users, self.num_users), ", Received: ", self.most_similar_users.shape)
            exit(-1)

        # item similarity matrix
        print("[CCF] load_similarity_matrices: loading item similarity matrix.....")
        self.item_similarity_matrix = np.load(item_file1)
        print("[CCF] load_similarity_matrices: item similarity matrix loading complete!")

        if self.item_similarity_matrix.shape != (self.num_items, self.num_items):
            print("[CCF] load_similarity_matrix: item similarity matrix dimensions mismatch. Expected: ",
                  (self.num_items, self.num_items), ", Received: ", self.item_similarity_matrix.shape)
            exit(-1)

        # most similar item matrix
        print("[CCF] load_similarity_matrices: loading most similar item matrix.....")
        self.most_similar_items = np.load(item_file2)
        print("[CCF] load_similarity_matrices: most similar item matrix loading complete!")

        if self.most_similar_items.shape != (self.num_items, self.num_items):
            print("[CCF] load_similarity_matrix: most similar item matrix dimensions mismatch. Expected: ",
                  (self.num_items, self.num_items), ", Received: ", self.most_similar_items.shape)
            exit(-1)

    # PREDICT RATING
    # if j != user_i and self.ratings[j, item_a] != 0:
    #     item_b = item_a
    # elif j != user_i and self.ratings[j, item_a] == 0:
    #     # find most similar item rated by user j
    def predict_rating(self, u_i, i_a, k, num_similar_users):

        # i_a -> index of book a in our matrix, i_id -> book_id, i_b -> index of that book in our matrix
        # u_i -> index of user i in our matrix, u_j -> index of user b in our matrix
        sum_term = 0
        sum_term_k = 0
        i_b = i_a

        # calculate summation term
        # consider only top "num_similar_users" similar users for rating prediction
        for idx in range(1, num_similar_users + 1):
            u_j = self.most_similar_users[u_i, idx]
            # get most similar item rated by user j
            for i_b in self.most_similar_items[i_a]:
                # i_b = self.item_idx[i_id]
                # if user u_j has rated item i_b
                if self.train_ratings[u_j, i_b] != 0:
                    break

            avg_r_u_j = self.user_data[u_j]["sum_ratings"] / self.user_data[u_j]["num_ratings"]
            sum_term += self.user_similarity_matrix[u_i, u_j] * self.item_similarity_matrix[i_a, i_b] * (
                    self.train_ratings[u_j, i_b] - avg_r_u_j)
            sum_term_k += self.user_similarity_matrix[u_i, u_j] * self.item_similarity_matrix[i_a, i_b]

        # rating = self.user_data[u_i]["sum_ratings"] / self.user_data[u_i]["num_ratings"] + k * sum_term
        rating = self.user_data[u_i]["sum_ratings"] / self.user_data[u_i]["num_ratings"] + sum_term/sum_term_k
        return rating

    def predict(self, k, num_similar_users=100):
        f = open("predicted_ratings.csv", 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["user_id", "book_id", "rating", "predicted"])
        print("[CCF] predict: predicting ratings.....")
        for sample in self.test_data:
            user = self.user_idx[sample[0]]
            book = self.item_idx[sample[1]]
            pred = self.predict_rating(user, book, k, num_similar_users)
            print(sample[0], "\t", sample[1], "\t", sample[2], "\t", pred,
                  self.user_data[user]["sum_ratings"] / self.user_data[user]["num_ratings"])
            csv_writer.writerow([str(sample[0]), str(sample[1]), str(sample[2]), str(pred)])
        f.close()
        print("[CCF] predict: ratings predicted!")

    def evaluate(self, k, num_similar_users=None, num_samples=None):
        MAE = 0
        RMSE = 0
        # f = open("predicted_ratings.csv", 'w')
        # csv_writer = csv.writer(f)
        # csv_writer.writerow(["user_id", "book_id", "rating", "predicted"])

        if num_samples is None:
            num_samples = len(self.test_data)

        if num_similar_users is None:
            num_similar_users = self.num_users - 1

        sample_num = 0
        num_samples = 50000

        print("[CCF] evaluate: Evaluation started.....")
        print("USER \t\t ITEM \t\t Actual \t\t Predicted \t\t Avg")
        for sample in self.test_data[100001:150000]:
            user = self.user_idx[sample[0]]
            book = self.item_idx[sample[1]]
            pred = self.predict_rating(user, book, k, num_similar_users)

            sample_num += 1
            print(sample_num)

            # print(sample[0], "\t\t", sample[1], "\t\t", sample[2], "\t\t", pred, "\t\t",
            #       self.user_data[user]["sum_ratings"] / self.user_data[user]["num_ratings"])
            # csv_writer.writerow([str(sample[0]), str(sample[1]), str(sample[2]), str(pred)])
            MAE += abs(sample[2] - pred)
            RMSE += (sample[2] - pred) ** 2

        # f.close()
        MAE = MAE / num_samples  # len(self.test_data)
        # RMSE = (RMSE / len(self.test_data)) ** (1 / 2)
        sqreRMSE = RMSE / num_samples
        RMSE = (RMSE / num_samples) ** (1 / 2)
        print("[CCF] evaluate: Mean Absolute Error:", MAE)
        print("[CCF] evaluate: Root Mean Squared Error", RMSE)
        print("[CCF] evaluate: sqred Root Mean Squared Error", sqreRMSE)


if __name__ == "__main__":
    obj_ccf = CombinedCollaborativeFiltering(10000, 6176)
    obj_ccf.data_loader(train_file="new_ratings_train.csv",
                        test_file="new_ratings_test.csv",
                        embed_file="summary_embeddings",
                        embed_model_file="my_doc2vec_model",
                        books_summary_file="new_books.csv")

    # obj_ccf.fit(method=1, save_file=True, matrix=2)
    obj_ccf.load_similarity_matrix()
    # print(obj_ccf.item_similarity_matrix[:10, :10])
    # print(obj_ccf.user_similarity_matrix[:10, :10])
    # print(obj_ccf.train_ratings[2, :])
    # obj_ccf.predict(0.05)
    obj_ccf.evaluate(0.0015)

# k = 0.0015, 9999, 1000
# Mean Absolute Error: 0.7062579345432203
# Root Mean Squared Error 0.9139636235938533

# no k, 9999, 1000
# Mean Absolute Error: 0.7068023733905398
# Root Mean Squared Error 0.9147857146439804

# no k, 1000, 1000
# Mean Absolute Error: 0.7022465276794804
# Root Mean Squared Error 0.9111865049593912

