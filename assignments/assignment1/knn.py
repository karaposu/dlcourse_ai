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
            dists = self.compute_distances_two_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

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
            v1=X[i_test]
            
            v1=v1.reshape(1, v1.shape[0])
            tmp=self.train_X-v1
            dists[i_test]=np.sum(np.abs(tmp),-1)
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
        from scipy.spatial import distance
        distance.euclidean([1, 0, 0], [0, 1, 0])  
        '''
        print("num_train:",num_train)
        print("num_test:",num_test)
        print("X.shape:",X.shape)
      
        print("train_X.shape:",self.train_X.shape)
        '''
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        dist = np.sqrt(np.sum(np.square(X[:,np.newaxis,:] - self.train_X), axis=2))
        # dists = -2 * np.dot(X, self.train_X.T) + np.sum(self.train_X**2,    axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
        return dists

    #  def find_nearest(array, value):
    #     array = np.asarray(array)
    #     idx = (np.abs(array - value)).argmin()
    #     return array[idx]

        

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
        
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            # self.train_X has the images  (16, 3072)
            # self.train_y has the the labels (121,)
            # dists array has array list of the euklodian distances of shape: (16, 121)
            
            
            # ind_of_train_example_with_min_distance= np.argmin(dists[i]) 
            # ind_of_train_example_with_min_distance
            # print("ind_of_train_example_with_min_distance for test_X[",i,"]:",ind_of_train_example_with_min_distance)
            # # self.train_y[i]

            # pred[i] =(self.train_y[ np.argmin(dists[i]) ] )
            

            # # self.find_nearest(dists,self.train_X[i])
            
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
            # nearest training samples
            pass
        return pred
