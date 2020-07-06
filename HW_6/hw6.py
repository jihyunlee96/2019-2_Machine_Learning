import os
import gzip
import time
import six.moves.cPickle as pickle
import numpy as np
from numpy.linalg import pinv
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier

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


class LDA:
	
	# initialize the dataset, labelset
	def __init__(self, dataset, labelset):
		self.dataset, self.labelset = dataset, labelset;

		# number of classes of the data
		self.num_class = np.max(labelset) - np.min(labelset) + 1

		# number of samples in the dataset
		self.num_sample = dataset.shape[0]

		# number of features in each data sample
		self.num_feature = dataset.shape[1]

		# calculate the weight vectors
		self._calculate_weight()

	
	# calculate the weight vectors
	def _calculate_weight(self):
	
		# 1. calculate within-class covariance (Sw)	

		class_mean = np.zeros((self.num_feature, self.num_feature))
		s_within = np.zeros((self.num_feature, self.num_feature))

		for c in range(self.num_class):

			# calculate the class mean
			idx = (c == self.labelset)
			class_samples = self.dataset[np.where(idx)]
			class_mean[c] = np.mean(class_samples, axis=0)
			
			# calculate within-class covariance for each class
			for n in range(class_samples.shape[0]):
				diff = np.subtract(class_samples[n], class_mean[c]) 
				s_within += np.outer(diff, diff.T)

		# 2. calculate between-class covariance (Sb)

		# calculate the global mean
		global_mean = np.mean(self.dataset, axis=0)
		s_between = np.zeros((self.num_feature, self.num_feature))	

		for c in range(self.num_class):

			# calculate the number of samples in class
			num_samples = np.sum(c == self.labelset)

			# calculate the between_class covariance for each class
			diff = np.subtract(class_mean[c], global_mean)

			s_between += num_samples * np.outer(diff, diff.T)

		#3. Find the eigenvectors of (Sw^-1)(Sb)
		mat = np.matmul(pinv(s_within), s_between)

		eigenvalue, eigenvector = np.linalg.eig(mat)

		# sort eigenvectors in descending order, with respect to eigenvalues
		indexes = list(range(len(eigenvalue)))
		indexes.sort(key=eigenvalue.__getitem__, reverse=True)

		eigenvalue = list(map(eigenvalue.__getitem__, indexes))

		for i, sublist in enumerate(eigenvector):
			eigenvector[i] = [sublist[j] for j in indexes]				
		
		self.weight = eigenvector

	# project the given dataset on the reduced space
	def project(self, dataset, dim):
		
		return np.matmul(dataset, self.weight[:,:dim])


def visualize(test_x, test_y, name):
	
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown', 'olive', 'plum']	
	
	for i in range(10):
		idx = (test_y == i)
		samples = test_x[np.where(idx)]	

		plt.plot(samples[:, 0], samples[:, 1], c=colors[i], marker='.', linestyle='',markersize=0.5)
	if 'lda' in name:
		plt.axis([-0.008, 0.008, -0.1, 0.1])
	plt.savefig(name) 


class PCA:
	
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
		print(eigenvalue[:10])
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
	
	dims = [2, 3, 5, 9]

	lda = LDA(train_x, train_y)

	for dim in dims:

		# LDA projection on train dataset
		lda = LDA(train_x, train_y)
		projected_data = lda.project(train_x, dim)

		# initialize and train the random forest classifier
		clf = RandomForestClassifier()
		clf.fit(projected_data, train_y)

		# LDA projection on test dataset & random forest prediction
		projected_test = lda.project(test_x, dim)
		result = clf.predict(projected_test)

		print('lda accuracy:', np.sum(result == test_y) / test_x.shape[0])

		# PCA projection on train dataset
		pca = PCA(train_x, dim)
		projected_data = pca.project(train_x)

		# initialize and train the random forest classifier
		clf = RandomForestClassifier()
		clf.fit(projected_data, train_y)

		# PCA projection on test dataset & KNN prediction
		projected_test = pca.project(test_x)
		result = clf.predict(projected_test)

		print('pca accuracy:', np.sum(result == test_y) / test_x.shape[0])
		
	'''
	dims = [2, 3, 5, 9]

	lda = LDA(train_x, train_y)
	for dim in dims:
		projected_data = lda.project(train_x, dim)

		clf = RandomForestClassifier(max_depth=10)
		clf.fit(projected_data, train_y)

		projected_test = lda.project(test_x, dim)

		result = clf.predict(projected_test)

		#visualize(projected_test, test_y, 'truth_lda.png')
		visualize(projected_test, result, 'lda.png')

		print('lda:', np.sum(result == test_y) / test_x.shape[0])

		pca = PCA(train_x, dim)
		projected_data2 = pca.project(train_x)

		#visualize(projected_data, train_y)

		clf2 = RandomForestClassifier(n_estimators=50)
		clf2.fit(projected_data2, train_y)

		projected_test2 = pca.project(test_x)

		result = clf2.predict(projected_test2)
		#visualize(projected_test2, test_y, 'truth_pca.png')
		#visualize(projected_test2, result, 'pca.png')

		print('pca:', np.sum(result == test_y) / test_x.shape[0])
	'''	
