import os
import numpy as np
import gzip
import six.moves.cPickle as pickle
import matplotlib.pylab as plt
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


def generate_mean_img(train_x):

    # calculate the mean vector
    mean = train_x.mean(0)

    # reshape the vector
    tmp_img = mean.reshape((28, 28))

    # rescale the matrix
    tmp_img /= tmp_img.max()/255

    # save the matrix into a file
    tmp_img = Image.fromarray(tmp_img.astype(np.uint8))
    tmp_img.save('mean.jpg')


def generate_variance_img(train_x):

    # calculate the variance vector    
    var_img = train_x.var(0)

    # reshape the vector
    var_img = var_img.reshape((28, 28))

    # rescale the matrix
    var_img /= var_img.max()/255

    # save the matrix into a file
    var_img = Image.fromarray(var_img.astype(np.uint8))
    var_img.save('var.jpg')


def calculate_eigenvalue_eigenvector(train_x):
    
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

    return eigenvalue, eigenvector.T


def generate_eigenvector_images(eigenvector):
    # NOTE: given eigenvectors are already sorted in descending order

    # discard the imaginary parts of eigenvectors 
    eigenvector = np.real(eigenvector)

    for i in range(10):

        # reshape the vector
        tmp_img = eigenvector[i].reshape((28, 28))

        # rescale the matrix
        tmp_img -= min(eigenvector[i])
        tmp_img /= tmp_img.max()/255

        # save the matrix into a file
        tmp_img = Image.fromarray(tmp_img.astype(np.uint8))
        tmp_img.save('eig_v_large'+str(i)+'.jpg')


def plot_eigenvalue_graph(eigenvalue):
    # NOTE: given eigenvalues are already sorted in descending order    

    x = list(range(100))
    y = eigenvalue[:100]

    plt.plot(x, y)

    plt.xlabel('index')
    plt.ylabel('eigenvalue')

    plt.savefig('eig_val.png')


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set

    '''
    generate_mean_img(train_x)

    generate_variance_img(train_x)

    eigenvalue, eigenvector = calculate_eigenvalue_eigenvector(train_x)

    generate_eigenvector_images(eigenvector)

    plot_eigenvalue_graph(eigenvalue)

	'''


