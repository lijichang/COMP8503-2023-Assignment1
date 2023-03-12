import os.path
import pickle
import numpy as np
from time import time
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10, SVHN, MNIST
from sklearn.manifold import TSNE
import umap
from utils import evaluate_output
from Isomap import isomap


def data_prep(data_path, dataset='MNIST', size=10000):
    '''
    This function loads the dataset as numpy array.
    Input:
        data_path: path of the folder you store all the data needed.
        dataset: the name of the dataset.
        size: the size of the dataset. This is useful when you only
              want to pick a subset of the data
    Output:
        X: the dataset in numpy array
        labels: the labels of the dataset.
    '''
    if dataset == 'MNIST':
        mnist = MNIST(root=data_path, download=True, train=True)
        X = mnist.train_data.reshape((len(mnist.train_data), -1))
        labels = mnist.train_labels.reshape((len(mnist.train_data), -1))
        X = X.numpy()
        labels = labels.numpy()
    elif dataset == 'CIFAR-10':
        cifar10 = CIFAR10(root=data_path, download=True, train=True)
        X = cifar10.train_data.reshape((len(cifar10.train_data), -1))
        labels = cifar10.train_labels
        labels = np.array(labels).reshape((len(cifar10.train_data), -1))
    elif dataset == 'SVHN':
        svhn = SVHN(root=data_path, download=True, split='train')
        X = svhn.data.reshape((len(svhn.data), -1))
        labels = svhn.labels
        labels = np.array(labels).reshape((len(svhn.data), -1))
    else:
        print('Unsupported dataset')
        assert(False)
    
    num_class = np.max(labels)
    num_samples_per_class = size // (num_class+1)
    selected_data = []
    selected_labels = []
    for i in range(num_class+1):
        class_indices = np.where(labels == i)[0]
        class_data = X[class_indices]
        selected_data.append(class_data[:num_samples_per_class])
        selected_labels.append(np.full(num_samples_per_class, i))
    X = np.concatenate(selected_data, axis=0)
    labels = np.concatenate(selected_labels, axis=0)

    return X, labels

def experiment(X, method='t-SNE', **kwargs):
    if method == 't-SNE':
        transformer = TSNE(**kwargs)
        start_time = time()
        X_low = transformer.fit_transform(X)
        total_time = time() - start_time
        print("This run's time:")
        print(total_time)
    elif method == 'UMAP':
        transformer = umap.UMAP(**kwargs)
        start_time = time()
        X_low = transformer.fit_transform(X)
        total_time = time() - start_time
        print("This run's time:")
        print(total_time)
    elif method == 'ISOMAP':
        start_time = time()
        X_low = isomap(X=X, n_neighbors=kwargs['n_neighbors'])
        total_time = time() - start_time
        print("This run's time:")
        print(total_time)
    else:
        print("Incorrect method specified")
        assert(False)

    return X_low, total_time


def main(data_path, output_path, dataset_name='MNIST', size=10000000):
    print("dataset_name", dataset_name)
    X, labels = data_prep(data_path, dataset=dataset_name, size=size) #size
    if X.shape[1] > 100:
        pca = PCA(n_components=100)
        X = pca.fit_transform(X)
    print("Data loaded successfully")

    # do experiment
    methods = ['ISOMAP', 't-SNE', 'UMAP']
    args = {
        't-SNE': [{'perplexity': 10}, {'perplexity': 20}, {'perplexity': 40}],
        'UMAP': [{'n_neighbors': 10}, {'n_neighbors': 20}, {'n_neighbors': 40}],
        'ISOMAP': [{'n_neighbors': 10}, {'n_neighbors': 20}, {'n_neighbors': 40}],
    }

    print("Experiment started")
    all_results = {}
    for method in methods:
        parameters = args[method]
        for parameter in parameters:
            X_low, total_time = experiment(X, method, **parameter)
            param1 = list(parameter.keys())[0]
            param2 = parameter[param1]
            cur_name = '{dataset_name}_{method}_{param1}_{param2}_{size}'.format(dataset_name=dataset_name, method=method, param1=param1, param2=param2, size=size)
            print("cur_name", cur_name)
            X_low = X_low.reshape((X.shape[0], -1))
            cur_name = output_path + cur_name
            np.save(cur_name+"-original_features", X)
            np.save(cur_name+"-low_features", X_low)
            np.save(cur_name+"-labels", labels)

    with open(dataset_name, 'wb') as fp:
        pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finished')

    return 0

if __name__ == '__main__':
    # Please define the data_path and output_path here
    data_path = "./data/"
    output_path = "./output/"

    main(data_path, output_path, 'MNIST', 20000)
    main(data_path, output_path, 'SVHN', 20000)
    main(data_path, output_path, 'CIFAR-10', 20000)
