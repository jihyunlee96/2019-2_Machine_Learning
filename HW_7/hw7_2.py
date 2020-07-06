import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder

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
		self._define_basis_func_random()

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

		return preds, np.square(pred - test_y).mean()



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
	xlsx_data = pandas.read_excel(file_name, index_col=None)
	xlsx_data.to_csv(file_name + '.csv', index=False, header=None, encoding='utf-8')

	file_name = file_name + '.csv'

	train_data = []

	with open(file_name) as file_data:
		train_data = file_data.readlines()

	for row_idx in range(len(train_data)):
		train_data[row_idx] = train_data[row_idx].split(',')

	train_data = np.array(train_data)

	train_data = np.where(train_data == '-', '0', train_data)
	train_data = np.where(train_data == '', '0', train_data)

	train_data= np.array(train_data)
	print('before', train_data[2])
	#train_data = np.delete(train_data, [0, 1, 4, 24], 1)
	print('col', train_data[:,0])

	for c in [0, 1, 4, 24]:
		lbl = LabelEncoder()
		lbl.fit(train_data[:,c])
		train_data[:,c] = lbl.transform(train_data[:,c])


	train_data = np.core.defchararray.strip(train_data)

	train_data = train_data.astype(np.float32)

	print('after', train_data[2])
	train_x = np.hstack((train_data[:, :2], train_data[:, 3:]))
	train_y = train_data[:, 2]

	return train_x, train_y

class linearRegression(torch.nn.Module):
	def __init__(self, inputSize, outputSize):
		super(linearRegression, self).__init__()
		self.linear1 = torch.nn.Linear(inputSize, 100)
		self.bn1 = nn.BatchNorm1d(num_features=100)
		self.linear2 = torch.nn.Linear(100, 300)
		self.bn2 = nn.BatchNorm1d(num_features=300)
		self.linear3 = torch.nn.Linear(300, 100)
		self.bn3 = nn.BatchNorm1d(num_features=100)
		self.linear4 = torch.nn.Linear(100, outputSize)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = self.bn1(x)
		x = F.relu(self.linear2(x))
		x = self.bn2(x)
		x = F.relu(self.linear3(x))
		x = self.bn3(x)
		x = self.linear4(x)
		return x



if __name__ == '__main__':

	x, y = load_data("data_set_train.xlsx")

	#train_transformer = Normalizer().fit(x)
	#x = train_transformer.transform(x)

	#mean_y = np.mean(y)

	x_max = np.max(x, axis=1)
	x_min = np.min(x, axis=1)

	for i in range(x.shape[1]):
		x[:,i] = (x[:,i] - x_min[i]) / (x_max[i] - x_min[i])

	y_max = np.max(y)
	y_min = np.min(y)
	y = (y - y_min) / (y_max - y_min)

	print('x_min: ', np.min(x))
	print('x_max: ', np.max(x))
	print('y_min: ', np.min(y))
	print('y_max: ', np.max(y))

	#x = preprocessing.normalize(x)

	#y = preprocessing.normalize(y.reshape(1, -1))
	#y = y.reshape(-1)

	print(x)
	print(y)

	train_x, test_x, train_y, test_y = train_test_split(
		x, y, test_size=0.1, random_state=42)
	#train_x2, train_y2 = load_data("RBFN_train_test_files/fa_train2.txt")

	#train_x = np.vstack((train_x, train_x2))
	#train_y = np.append(train_y, train_y2)


	#test_x, test_y = load_data("RBFN_train_test_files/fa_test.txt")
	print(train_x.shape)
	print(train_y.shape)

	#train_x = preprocessing.normalize(train_x)
	#test_x = preprocessing.normalize(test_x)

	pca = PCA(.95)
	pca.fit(train_x)
	train_x = pca.transform(train_x)
	test_x = pca.transform(test_x)
	
	#'''

	epochs = 2000
	model = linearRegression(train_x.shape[1], 1)
	criterion = torch.nn.MSELoss()
	#optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(epochs):

		inputs = torch.from_numpy(train_x)
		labels = torch.from_numpy(train_y)

		optimizer.zero_grad()

		outputs = model(inputs)

		loss = criterion(outputs, labels)

		loss.backward()

		# update parameters
		optimizer.step()

		print('epoch {}, loss {}'.format(epoch, loss.item()))

	#model.load_state_dict(torch.load('./20000.pth'))
	#print(model.eval())


	inputs = torch.from_numpy(test_x)
	pred_y = model(inputs)
	print("predict (after training)")
	print(pred_y[:10].detach().numpy().reshape(-1))
	print(test_y[:10])

	inputs = torch.from_numpy(train_x)
	pred_y = model(inputs)
	print()
	print(pred_y[:10].detach().numpy().reshape(-1) * y_max)
	print(train_y[:10] * y_max)


	torch.save(model.state_dict(), './model.pth')

	#'''
	#rbf = RBF(train_x, train_y, 50)
	#rbf.train()
	#preds, acc = rbf.evaluate_f(train_x, train_y)

	

	mse = (np.square(pred_y.detach().numpy().reshape(-1) - train_y)).mean(axis=None)
	print('mse: ', mse)
	print('mse: ', mse*y_max)
	'''
	nn = MLPRegressor(
		solver='adam',
		hidden_layer_sizes=1000,
		max_iter=200,
		shuffle= True,
		random_state=9876,
		learning_rate = 'adaptive',
		alpha=0.01,
		activation='relu')

	nn.fit(test_x, test_y)



	print(nn.score(train_x, train_y))
	print(nn.score(test_x, test_y))

	pred = nn.predict(test_x[:20])

	for i in range(20):
		print(pred[i], " -- ", test_y[i])

	'''
	'''
	pca = PCA(.95)
	pca.fit(train_x)
	train_x = pca.transform(train_x)

	rbf = RBF(train_x, train_y, 50)
	rbf.train()
	preds, acc = rbf.evaluate_f(train_x, train_y)
	'''

	#print(train_y)
	#print(preds)

	#print('acc.: ', acc)
