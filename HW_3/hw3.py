import os
import gzip
import time
import six.moves.cPickle as pickle
import numpy as np
from scipy import stats
import scipy.spatial
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    return train_set, valid_set, test_set


class KNN:
	
	# initialize the dataset, labelset, k value
	def __init__(self, dataset, labelset, k):
		self.dataset, self.labelset = dataset, labelset;
		self.k = k

	# calculate the distance matrix between the dataset and the given sample(s)
	def calculate_dist(self, test_sample, metric):
		distance_mat = scipy.spatial.distance.cdist(test_sample, self.dataset, metric)

		return distance_mat # N X 50000

	# apply majority vote algorithm
	def majority_vote(self, distance_mat):

		# sort the labelset with respect to corresponding distances
		idx = list(range(distance_mat.shape[0]))
		distances = list(distance_mat)
		idx.sort(key=distances.__getitem__)

		sorted_labels = list(map(self.labelset.__getitem__, idx))

		# only consider the labels of k nearest neighbors
		sorted_labels = sorted_labels[:self.k]

		# return the mode of the labels of k nearest neighbors
		result = stats.mode(sorted_labels).mode[0]
	
		return result

	# model prediction 
	def predict(self, test_sample, metric='euclidean'):

		# calculate the distances between the test sample and the dataset
		distance_mat = self.calculate_dist(test_sample, metric)

		# apply majority vote algorithm using the calculated distances
		result = self.majority_vote(distance_mat.T)

		return result

	# model evaluation
	def evaluate(self, eval_x, eval_y, metric='euclidean'):

		# calculate the distance matrix
		distance_mat = self.calculate_dist(eval_x, metric)

		correct_count = 0

		# apply majority vote algorithm for each evaluation sample,
		# and append the result to the list
		for i in range(distance_mat.shape[0]):
			result = self.majority_vote(distance_mat[i])

			if result == eval_y[i]:
				correct_count += 1

		# calculate the model accuracy
		accuracy = correct_count / distance_mat.shape[0]

		# print the evaluation results
		print('Evaluation based on ', eval_x.shape[0], 'samples: ')
		print('  - Accuracy: ', accuracy)


class Projection_on_Eigenspace:
	
	# calculate the eigenspace matrix
	def __init__(self, dataset, dim):

		# calculate the covariance matrix
		cov = np.cov(train_x.T)

		# calculate eigenvalue, eigenvector of the covariance matrix
		eigenvalue, eigenvector = np.linalg.eig(cov)

		# sort them in descending order, with respect to eigenvalues
		indexes = list(range(len(eigenvalue)))
		indexes.sort(key=eigenvalue.__getitem__, reverse=True)

		eigenvalue = list(map(eigenvalue.__getitem__, indexes))

		for i, sublist in enumerate(eigenvector):
			eigenvector[i] = [sublist[j] for j in indexes]

		# only consider the first n eigenvectors as eigenspace
		self.eigenspace = eigenvector[:, :dim]


	# project the given sample(s) onto n-dim eigenspace
	def project(self, samples):

		projected_samples = np.dot(samples, self.eigenspace)
			
		return projected_samples


if __name__ == '__main__':
	
	# load MNIST data
	train_set, val_set, test_set = load_data('mnist.pkl.gz')

	train_x, train_y = train_set
	val_x, val_y = val_set
	test_x, test_y = test_set
	
	knn_classifier = KNN(train_x, train_y, 1)

	for i in range(3):
		prediction = knn_classifier.predict(test_x[i].reshape(1, -1))
		print('Prediction for test[' + str(i) + '] : ', prediction)
	

	
	for i in range(3):
		tmp_img = test_x[i].reshape((28, 28))
		tmp_img *= 255.9
		tmp_img = Image.fromarray(tmp_img.astype(np.uint8))
		tmp_img.save('val' + str(i) + '.jpg')

	#knn_classifier.evaluate(val_x, val_y)

	'''
	#KDTree = scipy.spatial.KDTree(train_x)
	#print(KDTree.data)

	#eigenspace_projection = Projection_on_Eigenspace(train_x, 10)

	#projected_train_x = eigenspace_projection.project(train_x)

	#projected_train_x = normalize_data(projected_train_x)# for test

	#knn_classifier2 = KNN(projected_train_x, train_y, 5)
	
	#projected_val_x = eigenspace_projection.project(val_x)
	
	#projected_val_x = normalize_data(projected_val_x) # for test

	#knn_classifier2.evaluate(projected_val_x, val_y)
	'''


	'''
	for i in range(0, test_x.shape[0]):
		test_sample = np.reshape(test_x[i], (-1, 1)).T
		result = knn_classifier.predict(test_sample)
		if (result == test_y[i]): print('accurate!')
		print(result)
	'''
