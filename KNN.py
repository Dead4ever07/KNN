import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(x1-x2)**2)
    return distance

class KNN:
    def __init__(self, k=3):
        self.k =k
        

    def fit(self , X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predictions = [self.predict_(x) for x in X]
        return predictions


    def predict_(self, x):
        #compute the distance
        distance =[euclidean_distance(x,x_train) for x_train in self.X_train]

        #get the closest k

        k_indicies = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]

        #majority voye
        most_comon = Counter(k_nearest_labels).most_common()
        return most_comon[0][0]

