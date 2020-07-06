import numpy as np
import matplotlib.pyplot as plt

class KMeans:

	# initialize instance variables
	def __init__(self, dataset, k):
	
		self.dataset  = dataset
		self.predictions = np.zeros(self.dataset.shape[0])
		self.feature_dim = self.dataset.shape[1]
		self.k = k

		self.means = np.zeros((k, self.feature_dim))

		# randomly initialize mean vectors
		for i in range(self.k):
			
			# mean of randomly selected 20 data samples
			self.means[i] = np.mean(self.dataset[np.random.choice
                (self.dataset.shape[0], 1, replace=False)], axis=0)
 

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
			print(np.sum(prev_predictions != self.predictions))

		return self.predictions
        		

	# visualize the prediction result
	def plot(self, org_train_x):
		
		# project dataset on 2D eigenspace
		if self.feature_dim != 2:
			projection = Projection_on_Eigenspace(org_train_x, 2)
			projected_dataset = projection.project(org_train_x)
		
		else:
			projected_dataset = self.dataset

		# plot the dataset
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown', 'olive', 'plum']

		for i in range(self.k):
			idx = (self.predictions == i)
			samples_in_cluster = projected_dataset[np.where(idx)]

			# plot the data in the same cluster with the same color
			plt.plot(samples_in_cluster[:,0], samples_in_cluster[:,1], 
                                c=colors[i], marker='.', markersize=0.5)

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



class RBF:

	# class initialization
	def __init__(self, train_x, train_y, num_basis):

		self.train_x, self.train_y = train_x, train_y
		self.num_basis = num_basis		

		self.basis_mean = np.zeros((num_basis, train_x.shape[1]))
		self.basis_var = np.zeros((num_basis))

		# define the basis functions (mean and variances)
		self._define_basis_func_kmeans()	

		self.weight = np.zeros(train_x.shape[1])


	# define the basis functions using kmeans
	def _define_basis_func_kmeans(self):
		
		# apply kmeans algorithm with (k = num_basis)
		kmeans = KMeans(self.train_x, self.num_basis)
		cluster_result = kmeans.cluster()

		# for each cluster, calculate mean and variance to define basis functions
		for cluster_idx in range(self.num_basis):

			samples_idx = (cluster_result == cluster_idx)
			samples_in_cluster = self.train_x[np.where(samples_idx)]

			self.basis_mean[cluster_idx] = np.mean(samples_in_cluster, axis=0)
			self.basis_var[cluster_idx] = np.var(samples_in_cluster)


	# define the basis functions randomly
	def _define_basis_func_random(self):
	
		for cluster_idx in range(self.num_basis):
			self.basis_mean = np.random.rand(self.num_basis, self.train_x.shape[1])
			self.basis_var = np.random.rand(self.num_basis)
	

	# basis function (Gaussian kernel)
	def _basis_func(self, x, basis_idx):
		 return np.exp(-1 * np.power(np.linalg.norm(x - self.basis_mean[basis_idx]), 2)
			/ (2 * self.basis_var[basis_idx]))

	# train
	def train(self):
		
		# calculate the basis function matrix
		basis_fun_mat = np.zeros((self.train_x.shape[0], self.num_basis))
		
		for sample_idx in range(self.train_x.shape[0]):
			for basis_idx in range(self.num_basis):
				basis_fun_mat[sample_idx][basis_idx] = self._basis_func(self.train_x[sample_idx], basis_idx)
	
		# calculate the pseudo-inverse of basis function matrix	
		inverse = np.linalg.pinv(basis_fun_mat)

		# multiply inverse matrix and label vector to obtain the weight
		self.weight = np.matmul(inverse, self.train_y)


		# evaluate the network accuracy
	def evaluate(self, test_x, test_y):

		correct = 0
		preds = np.zeros(test_x.shape[0])
		print(test_x.shape[0])
		for i in range(test_x.shape[0]):
			g = np.zeros(self.num_basis)
			
			for k in range(self.num_basis):
				g[k] = self._basis_func(test_x[i], k)

			pred = np.dot(g, self.weight)

			#if pred > 0.5: pred = 1
			#else: pred = 0

			preds[i] = pred

			#if pred == test_y[i]: correct += 1

		self.visualize(test_x, preds)
		return correct / test_x.shape[0]

	# evaluate the network MSE
	def evaluate_f(self, test_x, test_y):

		preds = np.zeros(test_x.shape[0])

		for i in range(test_x.shape[0]):
			g = np.zeros(self.num_basis)
			
			for k in range(self.num_basis):
				g[k] = self._basis_func(test_x[i], k)

			pred = np.dot(g, self.weight)

			preds[i] = pred

		return np.square(pred - test_y).mean()



	def visualize(self, test_x, pred):
	
		colors = ['b', 'y']	
	
		for i in range(2):
			idx = (pred == i)
			samples = test_x[np.where(idx)]	

			plt.plot(samples[:, 0], samples[:, 1], c=colors[i], marker='.', linestyle='',markersize=0.5)

		np.set_printoptions(threshold=np.inf)

		print(self.basis_mean)
		print(self.basis_var)
		print(self.weight)

		plt.savefig('test.png') 


	def visualize_f(self, test_x, test_y, pred):
	
		plt.plot(test_x, pred, marker='.',color='r', linestyle='',markersize=0.5)
		plt.plot(test_x, test_y, marker='.',color='b', linestyle='',markersize=0.5)
		plt.savefig('test.png') 



def load_data(file_name):
	# load train data
	train_data = []

	with open(file_name) as file_data:
		train_data = file_data.readlines()
   
	for row_idx in range(len(train_data)):
		train_data[row_idx] = train_data[row_idx].split() 

	train_data = np.array(train_data).astype(np.float32)
	#train_x = train_data[:, :2]
	#train_y = train_data[:, 2]
	train_x = train_data[:, :1]
	train_y = train_data[:, 1]


	return train_x, train_y


if __name__ == '__main__':
	
	train_file_name = "RBFN_train_test_files/fa_train1.txt"

	train_x, train_y = load_data(train_file_name)
	train_x2, train_y2 = load_data("RBFN_train_test_files/fa_train2.txt")

	#train_x = np.vstack((train_x, train_x2))
	#train_y = np.append(train_y, train_y2)


	test_x, test_y = load_data("RBFN_train_test_files/fa_test.txt")

	print(test_x.shape)	
	#print(train_y.shape)	

	#for i in range(5):
	print("# sample", train_x2.shape[0])
	rbf = RBF(train_x2, train_y2, 50)
	rbf.train()
	acc = rbf.evaluate_f(test_x, test_y)
	print('acc.: ', acc)

