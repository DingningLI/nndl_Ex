# MNIST using JAX, using grad

import jax.numpy as jnp
from jax import grad, jit, vmap, random, jvp
from jax.tree_util import tree_map

import numpy as np
from torch.utils import data
from torchvision import datasets

from jax.scipy.special import logsumexp

# ------------------------------- hyper parameter section -------------------------------
num_epochs = 30
batch_size = 64
n_targets = 10
learning_rate = 0.064

def sigmoid(z):
    return 1.0/(1.0+jnp.exp(-z))

def relu(z):
  return jnp.maximum(0, z)

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

key = random.PRNGKey(0)
def random_layer_params(m, n, key, scale):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key, scale):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

key, init_key = random.split(key)  # init_key used for initialization
sizes = [784, 30, 10]
params = init_network_params(sizes, init_key, 0.001)

def set_value(x, value):
    return jnp.full_like(x, value)   

# ------------------------------- data loading section -------------------------------
class FlattenAndCast: # see torchvision.transforms for other examples
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=FlattenAndCast()
)

test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=FlattenAndCast()
)

train_dataloader = data.DataLoader(training_data, shuffle=True, batch_size=batch_size, collate_fn=numpy_collate)
# test_dataloader = data.DataLoader(test_data, batch_size=batch_size, collate_fn=numpy_collate)

# train_images = np.array(training_data.data).reshape(len(training_data.data), -1)
# train_labels = one_hot(np.array(training_data.targets), n_targets)
test_images = jnp.array(test_data.data.numpy().reshape(len(test_data.data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(test_data.targets), n_targets)

# ------------------------------- Training section -------------------------------
def predict(params, a):
    for w, b in params:
        a = relu(jnp.dot(w, a) + b) # change the act_fun here
    return a

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return jnp.mean((preds - targets) * (preds - targets))

@jit
def update(params, inputs, target):
    grads = grad(loss)(params, inputs, target)
    return tree_map(lambda p, g: p - learning_rate * g / batch_size, params, grads)

# I use a loop here because I don't know how to do vmap_JVP;
# another good thing is that this matches with the physical implementation.
def wf_estimator(params, inputs, target, key, num): # weight perturbed forward-AD, following (Baydin et al., 2022);
    # num is the number of random vevtors chosen to calculate the average of JVP.
    f = lambda params: loss(params, inputs, target) # Isolate the function from the weight matrix to the predictions
    # create a pytree of all 0's to store the vector (actually a pytree) for estimated grad vec
    jvpv_vec = tree_map(lambda x: set_value(x, 0), params)
    key, *num_init_keys = random.split(key, num = num+1)
    for sub_init_key in num_init_keys:
        v = init_network_params(sizes, sub_init_key, 1.0) # update at each iteration
        jvp_val = jvp(f, (params,), (v,))[1]
        jvp_val_pytree = tree_map(lambda x: set_value(x, jvp_val), params)
        jvp_vec_ = tree_map(lambda x, y: x*y, jvp_val_pytree, v)
        jvpv_vec = tree_map(lambda x, y: x+y, jvpv_vec, jvp_vec_)
    jvpv_vec = tree_map(lambda x: x / num, jvpv_vec) # averaged over the numbers
    return key, jvpv_vec # return the pytree of grad vecs

def wf_update(params, inputs, target, key, num):
    new_key, es_grads = wf_estimator(params, inputs, target, key, num) # estimated grad vecs
    return new_key, tree_map(lambda p, g: p - learning_rate * g / batch_size, params, es_grads)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    for idx, (x, y) in enumerate(train_dataloader):
        y = one_hot(y, n_targets)
        params = update(params, x, y)
        if idx % 200 == 0:  # evaluation
            # train_acc = accuracy(params, train_images, train_labels)
            # print("Training set accuracy {}".format(train_acc))
            test_acc = accuracy(params, test_images, test_labels)
            test_loss = loss(params, test_images, test_labels)
            print("Test set accuracy {}, Test set loss {}".format(test_acc, test_loss))

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}\n-------------------------------")
#     for idx, (x, y) in enumerate(train_dataloader):
#         y = one_hot(y, n_targets)
#         key, params = wf_update(params, x, y, key, 100)
#         if idx % 10 == 0:  # evaluation
#             # train_acc = accuracy(params, train_images, train_labels)
#             # print("Training set accuracy {}".format(train_acc))
#             test_acc = accuracy(params, test_images, test_labels)
#             test_loss = loss(params, test_images, test_labels)
#             print("Test set accuracy {}, Test set loss {}".format(test_acc, test_loss))    
        