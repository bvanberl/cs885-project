#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import shutil
import math
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils import data
from torch.autograd import Variable
import torch.utils.data as Data
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
bce = torch.nn.BCEWithLogitsLoss(reduction='none')
bce3 =  torch.nn.BCELoss(reduction='none')

def mask_threshold(x):
	x = (x+0.5).int().float()
	return x
  
def label_cov(labels):
	cov = torch.from_numpy(np.cov(labels, rowvar = False)).to(device)
	return cov
 
def get_labelcov_prior(batchsize, cov):
	#print(cov)
	v = torch.zeros(batchsize, cov.size()[0], cov.size()[1])
	for i in range(batchsize):
		v[i] = cov
	mean = torch.zeros(batchsize, cov.size()[1])
	return mean, v
 
def vector_expand(v):
	V = torch.zeros(v.size()[0],v.size()[1],v.size()[1]).to(device)
	for i in range(v.size()[0]):
		for j in range(v.size()[1]):
			V[i,j,j] = v[i,j]
	return V
 
def block_matmul(a, b):
	return None
  
def multivariate_sample(m,cov):
	m = m.reshape(m.size()[0],4)
	z = torch.zeros(m.size())
	for i in range(z.size()[0]):
		z[i] = MultivariateNormal(m[i].cpu(), cov[i].cpu()).sample()
	return z.to(device)
  
def kl_multinormal_cov(qm,qv, pm, pv):
	KL = torch.zeros(qm.size()[0]).to(device)
	for i in range(qm.size()[0]):
		#print(torch.det(qv[i].cpu()))
		KL[i] = 0.5 * (torch.log(torch.det(pv[i])) - torch.log(torch.det(qv[i])) +
		torch.trace(torch.inverse(pv[i]))*torch.trace(torch.inverse(qv[i])) +
		torch.norm(qm[i])*torch.norm(pv[i], p=1))
	return KL
 
 
def conditional_sample_gaussian(m,v):
	#64*3*4
	sample = torch.randn(m.size()).to(device)
	z = m + (v**0.5)*sample
	return z

def condition_gaussian_parameters(h, dim = 1):
	#print(h.size())
	m, h = torch.split(h, h.size(1) // 2, dim=1)
	m = torch.reshape(m, [-1, 3, 4])
	h = torch.reshape(h, [-1, 3, 4])
	v = F.softplus(h) + 1e-8
	return m, v

def condition_prior(scale, label, dim):
	mean = torch.ones(label.size()[0],label.size()[1], dim)
	var = torch.ones(label.size()[0],label.size()[1], dim)
	for i in range(label.size()[0]):
		for j in range(label.size()[1]):
			mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
			mean[i][j] = torch.ones(dim)*mul
			var[i][j] = torch.ones(dim)*1
	return mean, var

 
def bce2(r, x):
	return x * torch.log(r + 1e-7) + (1 - x) * torch.log(1 - r + 1e-7)

################################################################################
# Please familiarize yourself with the code below.
#
# Note that the notation is
# argument: argument_type: argument_shape
#
# Furthermore, the expected argument_shape is only a guideline. You're free to
# pass in inputs that violate the expected argument_shape provided you know
# what you're doing
################################################################################

def sample_multivariate(cov, loc = None):
	# if loc == None:
	# 	loc = torch.zeros((cov.shape[0], cov.shape[0]))
	latent_code = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=cov, precision_matrix=None, scale_tril=None, validate_args=None)
	return latent_code

def get_covariance_matrix(A):
	# requirements: A must be torcj
	assert A.size()[1] == A.size()[2]
	I = torch.zeros(A.size()).to(device)
	i = torch.eye(n = A.size()[1]).to(device)
	for j in range(A.size()[0]):
		I[j] = torch.inverse(torch.mm(torch.t((A[j]-i)), (A[j]-i)))
	
	return I


def sample_gaussian(m, v):
	"""
	Element-wise application reparameterization trick to sample from Gaussian

	Args:
		m: tensor: (batch, ...): Mean
		v: tensor: (batch, ...): Variance

	Return:
		z: tensor: (batch, ...): Samples
	"""
	################################################################################
	# TODO: Modify/complete the code here
	# Sample z
	################################################################################

	################################################################################
	# End of code modification
	################################################################################
	sample = torch.randn(m.shape).to(device)
	

	z = m + (v**0.5)*sample
	return z

def log_normal(x, m, v):
	"""
	Computes the elem-wise log probability of a Gaussian and then sum over the
	last dim. Basically we're assuming all dims are batch dims except for the
	last dim.

	Args:
		x: tensor: (batch, ..., dim): Observation
		m: tensor: (batch, ..., dim): Mean
		v: tensor: (batch, ..., dim): Variance

	Return:
		kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
			that the summation dimension (dim=-1) is not kept
	"""
	################################################################################
	# TODO: Modify/complete the code here
	# Compute element-wise log probability of normal and remember to sum over
	# the last dimension
	################################################################################
	#print("q_m", m.size())
	#print("q_v", v.size())
	const = -0.5*x.size(-1)*torch.log(2*torch.tensor(np.pi))
	#print(const.size())
	log_det = -0.5*torch.sum(torch.log(v), dim = -1)
	#print("log_det", log_det.size())
	log_exp = -0.5*torch.sum( (x - m)**2/v, dim = -1)

	log_prob = const + log_det + log_exp

	################################################################################
	# End of code modification
	################################################################################
	return log_prob


def log_normal_mixture(z, m, v):
	"""
	Computes log probability of a uniformly-weighted Gaussian mixture.

	Args:
		z: tensor: (batch, dim): Observations
		m: tensor: (batch, mix, dim): Mixture means
		v: tensor: (batch, mix, dim): Mixture variances

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""
	################################################################################
	# TODO: Modify/complete the code here
	# Compute the uniformly-weighted mixture of Gaussians density for each sample
	# in the batch
	################################################################################
	z = z.unsqueeze(1)
	log_probs = log_normal(z, m, v)
	#print("log_probs_mix", log_probs.shape)

	log_prob = log_mean_exp(log_probs, 1)
	#print("log_prob_mix", log_prob.size())

	################################################################################
	# End of code modification
	################################################################################
	return log_prob


def gaussian_parameters(h, dim=-1):
	"""
	Converts generic real-valued representations into mean and variance
	parameters of a Gaussian distribution

	Args:
		h: tensor: (batch, ..., dim, ...): Arbitrary tensor
		dim: int: (): Dimension along which to split the tensor for mean and
			variance

	Returns:z
		m: tensor: (batch, ..., dim / 2, ...): Mean
		v: tensor: (batch, ..., dim / 2, ...): Variance
	"""
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v


def log_bernoulli_with_logits(x, logits):
	"""
	Computes the log probability of a Bernoulli given its logits

	Args:
		x: tensor: (batch, dim): Observation
		logits: tensor: (batch, dim): Bernoulli logits

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""
	log_prob = -bce(input=logits, target=x).sum(-1)
	return log_prob

def log_bernoulli_with_logits_nosigmoid(x, logits):
	"""
	Computes the log probability of a Bernoulli given its logits

	Args:
		x: tensor: (batch, dim): Observation
		logits: tensor: (batch, dim): Bernoulli logits

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""

	log_prob = bce2(logits, x).sum(-1)

	return log_prob




def kl_cat(q, log_q, log_p):
	"""
	Computes the KL divergence between two categorical distributions

	Args:
		q: tensor: (batch, dim): Categorical distribution parameters
		log_q: tensor: (batch, dim): Log of q
		log_p: tensor: (batch, dim): Log of p

	Return:
		kl: tensor: (batch,) kl between each sample
	"""
	element_wise = (q * (log_q - log_p))
	kl = element_wise.sum(-1)
	return kl


def kl_normal(qm, qv, pm, pv):
	"""
	Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
	sum over the last dimension

	Args:
		qm: tensor: (batch, dim): q mean
		qv: tensor: (batch, dim): q variance
		pm: tensor: (batch, dim): p mean
		pv: tensor: (batch, dim): p variance

	Return:
		kl: tensor: (batch,): kl between each sample
	"""
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl


def duplicate(x, rep):
	"""
	Duplicates x along dim=0

	Args:
		x: tensor: (batch, ...): Arbitrary tensor
		rep: int: (): Number of replicates. Setting rep=1 returns orignal x
  z 
	Returns:
		_: tensor: (batch * rep, ...): Arbitrary replicated tensor
	"""
	return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def log_mean_exp(x, dim):
	"""
	Compute the log(mean(exp(x), dim)) in a numerically stable manner

	Args:
		x: tensor: (...): Arbitrary tensor
		dim: int: (): Dimension along which mean is computed

	Return:
		_: tensor: (...): log(mean(exp(x), dim))
	"""
	return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
	"""
	Compute the log(sum(exp(x), dim)) in a numerically stable manner

	Args:
		x: tensor: (...): Arbitrary tensor
		dim: int: (): Dimension along which sum is computed

	Return:
		_: tensor: (...): log(sum(exp(x), dim))
	"""
	max_x = torch.max(x, dim)[0]
	new_x = x - max_x.unsqueeze(dim).expand_as(x)
	return max_x + (new_x.exp().sum(dim)).log()


def load_model_by_name(models_dir, model, global_step):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	file_path = os.path.join(models_dir, 'checkpoints',
							 model.name,
							 'model-{:05d}.pt'.format(global_step))
	state = torch.load(file_path)
	model.load_state_dict(state)
	print("Loaded from {}".format(file_path))


################################################################################
# No need to read/understand code beyond this point. Unless you want to.
# But do you tho ??\_(????_/??
################################################################################


def evaluate_lower_bound(model, labeled_test_subset, run_iwae=True):
	check_model = isinstance(model, VAE) or isinstance(model, GMVAE) or isinstance(model, LVAE)
	assert check_model, "This function is only intended for VAE and GMVAE"

	print('*' * 80)
	print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
	print('*' * 80)

	xl, _ = labeled_test_subset
	torch.manual_seed(0)
	xl = torch.bernoulli(xl)

	def detach_torch_tuple(args):
		return (v.detach() for v in args)

	def compute_metrics(fn, repeat):
		metrics = [0, 0, 0]
		for _ in range(repeat):
			niwae, kl, rec = detach_torch_tuple(fn(xl))
			metrics[0] += niwae / repeat
			metrics[1] += kl / repeat
			metrics[2] += rec / repeat
		return metrics

	# Run multiple times to get low-var estimate
	nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
	print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

	if run_iwae:
		for iw in [1, 10, 100, 1000]:
			repeat = max(100 // iw, 1) # Do at least 100 iterations
			fn = lambda x: model.negative_iwae_bound(x, iw)
			niwae, kl, rec = compute_metrics(fn, repeat)
			print("Negative IWAE-{}: {}".format(iw, niwae))


# def evaluate_classifier(model, test_set):
# 	check_model = isinstance(model, SSVAE)
# 	assert check_model, "This function is only intended for SSVAE"
#
# 	print('*' * 80)
# 	print("CLASSIFICATION EVALUATION ON ENTIRE TEST SET")
# 	print('*' * 80)
#
# 	X, y = test_set
# 	pred = model.cls.classify(X)
# 	accuracy = (pred.argmax(1) == y).float().mean()
# 	print("Test set classification accuracy: {}".format(accuracy))


def save_model_by_name(models_dir, model, global_step):
	save_dir = os.path.join(models_dir, 'checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
	log_dir = os.path.join('logs', model_name)
	save_dir = os.path.join('checkpoints', model_name)
	if overwrite_existing:
		delete_existing(log_dir)
		delete_existing(save_dir)
	# Sadly, I've been told *not* to use tensorflow :<
	# writer = tf.summary.FileWriter(log_dir)
	writer = None
	return writer


def log_summaries(writer, summaries, global_step):
	pass # Sad :<
	# for tag in summaries:
	#	 val = summaries[tag]
	#	 tf_summary = tf.Summary.Value(tag=tag, simple_value=val)
	#	 writer.add_summary(tf.Summary(value=[tf_summary]), global_step)
	# writer.flush()


def delete_existing(path):
	if os.path.exists(path):
		print("Deleting existing path: {}".format(path))
		shutil.rmtree(path)


def reset_weights(m):
	try:
		m.reset_parameters()
	except AttributeError:
		pass


def get_mnist_data(device, use_test_subset=True):
	preprocess = transforms.ToTensor()
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=True, download=True, transform=preprocess),
		batch_size=100,
		shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=False, download=True, transform=preprocess),
		batch_size=100,
		shuffle=True)

	# Create pre-processed training and test sets
	X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
	y_train = train_loader.dataset.train_labels.to(device)
	X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
	y_test = test_loader.dataset.test_labels.to(device)

	# Create supervised subset (deterministically chosen)
	# This subset will serve dual purpose of log-likelihood evaluation and
	# semi-supervised learning. Pretty hacky. Don't judge :<
	X = X_test if use_test_subset else X_train
	y = y_test if use_test_subset else y_train

	xl, yl = [], []
	for i in range(10):
		idx = y == i
		idx_choice = get_mnist_index(i, test=use_test_subset)
		xl += [X[idx][idx_choice]]
		yl += [y[idx][idx_choice]]
	xl = torch.cat(xl).to(device)
	yl = torch.cat(yl).to(device)
	yl = yl.new(np.eye(10)[yl])
	labeled_subset = (xl, yl)

	return train_loader, labeled_subset, (X_test, y_test)


def get_mnist_index(i, test=True):
	# Obviously *hand*-coded
	train_idx = np.array([[2732,2607,1653,3264,4931,4859,5827,1033,4373,5874],
						  [5924,3468,6458,705,2599,2135,2222,2897,1701,537],
						  [2893,2163,5072,4851,2046,1871,2496,99,2008,755],
						  [797,659,3219,423,3337,2745,4735,544,714,2292],
						  [151,2723,3531,2930,1207,802,2176,2176,1956,3622],
						  [3560,756,4369,4484,1641,3114,4984,4353,4071,4009],
						  [2105,3942,3191,430,4187,2446,2659,1589,2956,2681],
						  [4180,2251,4420,4870,1071,4735,6132,5251,5068,1204],
						  [3918,1167,1684,3299,2767,2957,4469,560,5425,1605],
						  [5795,1472,3678,256,3762,5412,1954,816,2435,1634]])

	test_idx = np.array([[684,559,629,192,835,763,707,359,9,723],
						 [277,599,1094,600,314,705,551,87,174,849],
						 [537,845,72,777,115,976,755,448,850,99],
						 [984,177,755,797,659,147,910,423,288,961],
						 [265,697,639,544,543,714,244,151,675,510],
						 [459,882,183,28,802,128,128,53,550,488],
						 [756,273,335,388,617,42,442,543,888,257],
						 [57,291,779,430,91,398,611,908,633,84],
						 [203,324,774,964,47,639,131,972,868,180],
						 [1000,846,143,660,227,954,791,719,909,373]])

	if test:
		return test_idx[i]

	else:
		return train_idx[i]


def get_svhn_data(device):
	preprocess = transforms.ToTensor()
	train_loader = torch.utils.data.DataLoader(
		datasets.SVHN('data', split='extra', download=True, transform=preprocess),
		batch_size=100,
		shuffle=True)

	return train_loader, (None, None), (None, None)


def gumbel_softmax(logits, tau, eps=1e-8):
	U = torch.rand_like(logits)
	gumbel = -torch.log(-torch.log(U + eps) + eps)
	y = logits + gumbel
	y = F.softmax(y / tau, dim=1)
	return y

class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[S??nderby 2016].
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n

	def __iter__(self):
		return self

	def __next__(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t

class FixedSeed:
	def __init__(self, seed):
		self.seed = seed
		self.state = None

	def __enter__(self):
		self.state = np.random.get_state()
		np.random.seed(self.seed)

	def __exit__(self, exc_type, exc_value, traceback):
		np.random.set_state(self.state)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def prune(A):
	zero = torch.zeros_like(A).to(device)
	A = torch.where(A < 0.3, zero, A)
	return A


def gumble_dag_loss(A):
	expm_A = torch.exp(F.gumbel_softmax(A))
	l = torch.trace(expm_A) - A.size()[0]
	return l


def filldiag_zero(A):
	mask = torch.eye(A.size()[0], A.size()[0]).byte().to(device)
	A.masked_fill_(mask, 0)
	return A


def matrix_poly(matrix, d):
	x = torch.eye(d).to(device) + torch.div(matrix.to(device), d).to(device)
	return torch.matrix_power(x, d)


def mask_threshold(x):
	x = (x + 0.5).int().float()
	return x


def matrix_poly(matrix, d):
	x = torch.eye(d).to(device) + torch.div(matrix.to(device), d).to(device)
	return torch.matrix_power(x, d)


def _h_A(A, m):
	expm_A = matrix_poly(A * A, m)
	h_A = torch.trace(expm_A) - m
	return h_A


def get_parse_args():
	# parse some given arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--every_degree', '-N', type=int, default=10,
						help='every N degree as a partition of dataset')
	args = parser.parse_args()
	return args


def weights_init(m):
	if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif (type(m) == nn.BatchNorm2d):
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif (type(m) == nn.Linear):
		nn.init.kaiming_uniform_(m.weight)
		nn.init.constant_(m.bias, 0)


def compute_theta_class(theta):
	# return: the class of theta where the theta fall into
	classes_num = 18
	interval = np.linspace(0, 360, num=classes_num)
	i = 0
	for start, end in zip(interval[:-1], interval[1:]):
		if theta == 360:
			return 17
		elif start <= theta < end:
			return i
		i = i + 1


class dataload(data.Dataset):
	def __init__(self, root):
		imgs = os.listdir(root)
		self.imgs = [os.path.join(root, k) for k in imgs]
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		img_path = self.imgs[idx]
		pil_img = Image.open(img_path)
		array = np.asarray(pil_img)
		data = torch.from_numpy(array)
		if self.transforms:
			data = self.transforms(pil_img)
		else:
			pil_img = np.asarray(pil_img).reshape(96, 96, 4)
			data = torch.from_numpy(pil_img)
		return data

	def __len__(self):
		return len(self.imgs)


def read_label(root, idl):
	with open(root, 'r') as f:
		reader = f.readlines()
		reader = [x.replace("  ", " ") for x in reader]
		reader = np.array([np.array(list(map(int, x[10:].strip().split(" ")))) for x in reader[2:]])
		reader = reader[:, idl]
		return reader[:200000]


class dataload_withlabel(data.Dataset):
	def __init__(self, root, dataset="train"):
		root = root + "/" + dataset

		imgs = os.listdir(root)

		self.dataset = dataset

		self.imgs = [os.path.join(root, k) for k in imgs]
		self.imglabel = [list(map(int, k[:-4].split("_")[1:])) for k in imgs]
		# print(self.imglabel)
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		# print(idx)
		img_path = self.imgs[idx]

		label = torch.from_numpy(np.asarray(self.imglabel[idx]))
		# print(len(label))
		pil_img = Image.open(img_path)
		array = np.asarray(pil_img)
		array1 = np.asarray(label)
		label = torch.from_numpy(array1)
		data = torch.from_numpy(array)
		if self.transforms:
			data = self.transforms(pil_img)
		else:
			pil_img = np.asarray(pil_img).reshape(96, 96, 4)
			data = torch.from_numpy(pil_img)

		return data, label.float()

	def __len__(self):
		return len(self.imgs)

class ExperienceDataset(data.Dataset):
	def __init__(self, root, xp_filename):

		self.xp_df = pd.read_csv(os.path.join(root, xp_filename))
		self.labels = self.xp_df.drop('Observation', axis=1).to_numpy()
		self.observation_paths = [os.path.join(root, k) for k in self.xp_df['Observation']]

		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		obs_path = self.observation_paths[idx]
		obs = np.load(obs_path)
		obs = np.moveaxis(obs, 0, -1) / 255.
		label = torch.from_numpy(self.labels[idx])

		if self.transforms:
			obs = self.transforms(obs)
		return obs.float(), label.float()

	def __len__(self):
		return len(self.observation_paths)


def get_partitions(every_n_degree):
	"""
    :param every_n_degree: every n degree as a partition
    :return: a list of intervals where training data should fall into
    """
	partitions_anchors_num = math.floor(360 / every_n_degree)
	partitions_anchors = np.linspace(0, partitions_anchors_num * every_n_degree, num=partitions_anchors_num + 1,
									 endpoint=True, dtype=int)
	if 360 % every_n_degree == 0:
		pass
	else:
		partitions_anchors = np.append(partitions_anchors, 360)

	partitions_list = []
	for start, end in zip(partitions_anchors[:-1], partitions_anchors[1:]):
		partitions_list.append([start, end])

	training_data_partitions = partitions_list[0::2]
	# test_data_partitions = partitions_list[1::2]
	return training_data_partitions


def whether_num_fall_into_intevals(number, intervals_list):
	"""
    :param number: given number need to determine
    :param intervals_list: a list including many intervals
    :return: boolean, whether the number will fall into one of the interval of the list,
    if falls, return True; otherwise, return False
    """
	for interval in intervals_list:
		if number >= interval[0] and number < interval[1]:
			return True
	if number == interval[1]:  # consider the most endpoint
		return True
	return False


def get_batch_unin_dataset_withlabel(dataset_dir, batch_size):
	dataset = ExperienceDataset(dataset_dir, 'xp_df.csv')
	dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return dataset
