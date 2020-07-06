import os
import gzip
import six.moves.cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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


# extract subarrays of train_x and train_y that correspond to 3 and 9 images
def extract_3_and_9_images(train_x, train_y):

	# extract indexes of 3 and 9 images
	idx_3 = (train_y == 3) 
	idx_9 = (train_y == 9)

	idx = np.logical_or(idx_3, idx_9)

	# extract subarrays of train_x and train_y based on the indexes
	train_x = train_x[np.where(idx)]
	train_y = train_y[np.where(idx)]

	return train_x, train_y


class KMeans:

	# initialize instance variables
	def __init__(self, dataset, labelset, k):
	
		self.dataset, self.labelset = dataset, labelset
		self.predictions = np.zeros(self.dataset.shape[0])
		self.feature_dim = self.dataset.shape[1]
		self.k = k

		self.means = np.zeros((k, self.feature_dim))

		# randomly initialize mean vectors
		for i in range(self.k):
			
			# mean of randomly selected 20 data samples
			self.means[i] = np.mean(self.dataset[np.random.choice(self.dataset.shape[0], 20, replace=False)], axis=0)
 

	# assignment step of KMeans algorithm	
	def assignment_step(self):

		distances = np.zeros(self.k)

		# iterate over each sample
		for sample_idx in range(self.dataset.shape[0]):

			# calculate the distance between the sample and mean vectors of k clusters
			for k_idx in range(self.k):
				distances[k_idx] = np.linalg.norm(self.dataset[sample_idx] - self.means[k_idx])

			# assign each sample to a cluster whose mean is the closest
			self.predictions[sample_idx] = np.argmin(distances)
		

	# update step of KMeans algorithm
	def update_step(self):

		# iterate over each cluster
		for i in range(self.k):
			idx = (self.predictions == i)
			self.means[i] = np.mean(self.dataset[np.where(idx)], axis=0)

	
	# cluster the dataset using KMeans algorithm
	def cluster(self):
		
		prev_predictions = np.ones(self.dataset.shape[0])
		step = 0

		# continue until the prediction doesn't change
		while (not np.array_equal(prev_predictions, self.predictions)):

			prev_predictions = np.copy(self.predictions)

			self.assignment_step()
			self.update_step()

			step += 1
			print(step)
			print(np.sum(prev_predictions - self.predictions))
			if (np.sum(prev_predictions - self.predictions) == 10089.0): 
				for i in range(self.k):
					idx = self.predictions == i
					print(self.dataset[np.where(idx)].shape[0])
				break

		return self.predictions
        		

	# visualize the prediction result
	def plot(self):
		
		# project dataset on 2D eigenspace
		if self.feature_dim != 2:
			projection = Projection_on_Eigenspace(self.dataset, 2)
			projected_dataset = projection.project(self.dataset)
		
		else:
			projected_dataset = self.dataset

		# plot the dataset
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown', 'olive', 'plum']

		for i in range(self.k):
			idx = (self.predictions == i)
			samples_in_cluster = projected_dataset[np.where(idx)]

			# plot the data in the same cluster with the same color
			plt.plot(samples_in_cluster[:,0], samples_in_cluster[:,1], c=colors[i], linestyle='',marker='.', markersize=0.5)

		plt.savefig('test.png')
		

	# visualize the mean vector of each cluster
	def mean(self):
        
		# iterate over each cluster
		for i in range(self.k):
			mean = self.means[i]

			# reshape and rescale the vector
			mean = mean.reshape((28, 28))
			mean /= mean.max() / 255

			# save mean as an image
			mean = Image.fromarray(mean.astype(np.uint8))
			mean.save('mean_' + str(i) + '.jpg')


    # calculate MSE
	def MSE(self):

		squared_error = 0
        
		# iterate over each cluster
		for i in range(self.k):
			idx = (self.predictions == i)
			samples_in_cluster = self.dataset[np.where(idx)]

            # iterate over each sample in current cluster
			for n in range(samples_in_cluster.shape[0]):

				# calculate SE 
				squared_error += np.sum(np.square(np.subtract(samples_in_cluster[n], self.means[i]))) / 784

        # return MSE
		return squared_error / self.dataset.shape[0]
			
    	

		
if __name__ == '__main__':
	
	# load MNIST data
	train_set, val_set, test_set = load_data('mnist.pkl.gz')

	train_x, train_y = train_set
	#val_x, val_y = val_set
	#test_x, test_y = test_set
	
	# extract subarrays of train_x and train_y that correspond to 3 and 9 images
	train_x, train_y = extract_3_and_9_images(train_x, train_y)

	# create an instance of KMeans class
	kmeans = KMeans(train_x, train_y, 10)
	
	preds = kmeans.cluster()	

	kmeans.plot()
	kmeans.mean()

	print('Squared error:', kmeans.MSE())



