import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd



class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.all_points = []
        self.labels = 0

    def compute_minkowski_dist(self, vec1, vec2, p):
        """
        computes minkowski distance between two vectors
        :param vec1: vector of features
        :param vec2: vector of features
        :param p: parameter for Minkowski distance calculation.
        :return: distance
        """
        len1 = len(vec1)
        sum1 = 0
        for i in range(len1):
            sum1 = sum1 + ((abs(vec1[i] - vec2[i])) ** p)
        return sum1 ** (1 / p)

    def find_neighboors(self, train, test_row, k):
        """
        find k nearest neighbors of the sample
        :return: lists of neighbors, and list of the distances
        """
        dist_list = []
        for sample in train:
            dist = self.compute_minkowski_dist(test_row, sample[0], self.p)
            dist_list.append((sample, dist))
        dist_list.sort(key=lambda tup: tup[1])
        neighboors, dist = [], []
        for i in range(k):
            neighboors.append(dist_list[i][0])
            dist.append(dist_list[i][1])
        return neighboors, dist

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        i = 0
        self.labels = set(y[:-1])
        for row in X:
            temp = []
            temp.append(row)
            label = y[i]
            temp.append(label)
            self.all_points.append(temp)
            i = i + 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        prediction1 = []
        for x in X:
            neighbors, dists_values = self.find_neighboors(self.all_points, x, self.k)
            output_values = [row[1] for row in neighbors]
            labels_dict = {}
            for neigh in output_values:
                if neigh in labels_dict.keys():
                    labels_dict[neigh] = labels_dict[neigh] + 1
                else:
                    labels_dict[neigh] = 1
            max_value = max(labels_dict.values())  # maximum value
            max_keys = [k for k, v in labels_dict.items() if
                        v == max_value]  # getting all keys containing the `maximum`
            len_max_keys = len(max_keys)  # number of ties between the labels
            if len_max_keys == 1:
                prediction1.append(max_keys[0])
            else:
                lexi_labels = []
                first_label_neigh = 0
                while neighbors[first_label_neigh] not in max_keys:
                    first_label_neigh += 1
                lexi_labels.append(neighbors[first_label_neigh])
                while dists_values[first_label_neigh] == dists_values[first_label_neigh + 1]:
                    if neighbors[first_label_neigh + 1] in max_keys:
                        lexi_labels.append(neighbors[first_label_neigh + 1])
                    first_label_neigh += 1
                if len(lexi_labels) == 1:
                    prediction1.append(lexi_labels[0])
                else:
                    lexi_labels.sort()
                    prediction1.append(lexi_labels[0])
        prediction = np.array(prediction1)
        return prediction



def main():
    print("*" * 20)
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()
    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)
    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
