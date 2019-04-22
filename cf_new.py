import numpy as np
import csv


class CombinedCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.ratings = np.zeros((num_users, num_items))  # num_users x num_items

    def load_rating_csv(self, rating_file):
        with open(rating_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)

        del data[0]
        print("[CCF] load_rating_csv: number of samples: ", len(data))

        # convert to int
        data = [list(map(int, i)) for i in data]
        for sample in data:
            self.ratings[sample[0], sample[1]] = sample[2]
        print("[CCF] load_rating_csv: rating matrix made")


if __name__ == "__main__":
    obj_ccf = CombinedCollaborativeFiltering(10000, 6176)
    obj_ccf.load_rating_csv("new_ratings.csv")
    print(obj_ccf.ratings[2, :])
