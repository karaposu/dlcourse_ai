
#deneme git yo
import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)
        # print("dists shape:", dists.shape)
        # print("dists:", dists)
        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                dists[i_test,i_train]=np.sum(np.abs([ X[i_test]- self.train_X[i_train]]))

            return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        '''
        print("X.shape:",X.shape)
        print("X[0] or v1:",X[0])
        print("X[0].shape:",X[0].shape)
        k=X[0].reshape(1, X[0].shape[0])
        print("v1.reshape(1, v1.shape[0]):",k)
        print("v1.reshaped to:",k.shape)
        '''
        
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions

            distance = np.abs(X[i_test] - self.train_X)
            distance = distance.sum(axis=1).reshape(1, -1)
            dists[i_test] = distance

        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]

        '''
        print("num_train:",num_train)
        print("num_test:",num_test)
        print("X.shape:",X.shape)
      
        print("train_X.shape:",self.train_X.shape)
        '''
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!

        dists =  np.sum(self.train_X , axis=1) - np.sum(X, axis=1)[:, np.newaxis]

        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        # print("dists:",dists)
        num_test = dists.shape[0]


        # 16




        pred = np.zeros(num_test, np.bool)




        # print("self.train_y:"  ,self.train_y)

        # print("k is:", self.k)
        for i in range(num_test):
            # print("dists[i]:",dists[i])
            # TODO: Implement choosing best class based on k
            # nearest training samples

            # self.train_X has the images  (16, 3072)
            # self.train_y has the the labels (121,)
            # dists array has array list of the l1 distances of shape: (16, 121)
            # we choose dist[0] which is first test images distanceses for all other train data points
            # now we will check the smallest n numbers
            # put them in a list.
            # find labels through these indexes in train labels
            # and see if total number of true is bigger than total number of false



            indexes_of_k_smallest_values= np.argpartition(dists[i], self.k)[:self.k]

            filter_indices = indexes_of_k_smallest_values
            axis = 0
            labels_of_k_smallest_values = np.zeros(self.k)

            labels_of_k_smallest_values = np.take(self.train_y, filter_indices, axis)
            # indexes_of_k_smallest_values = np.argpartition(dists[i], self.k)
            # print("indexes_of_k_smallest_values for test image[", i, "]:", indexes_of_k_smallest_values)
            
            

            # print("labels_of_k_smallest_values:"  ,labels_of_k_smallest_values)
            pred[i] = np.bincount(labels_of_k_smallest_values).argmax()



            # print("KNN prediction for test image[",i,"] is:",pred[i])
        self.prediction=  pred

        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k

            indexes_of_k_smallest_values = np.argpartition(dists[i], self.k)[:self.k]

            filter_indices = indexes_of_k_smallest_values
            axis = 0


            labels_of_k_smallest_values = np.take(self.train_y, filter_indices, axis)
            (values, counts) = np.unique(labels_of_k_smallest_values, return_counts=True)
            # print("values:", values)
            # print("counts:", counts)
            pred[i] = values[np.argmax(counts)]

        return pred





