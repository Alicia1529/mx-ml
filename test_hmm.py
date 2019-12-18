from sklearn import datasets
from numpy import loadtxt
from line_profiler import LineProfiler
from collections import Counter
import mxnet as mx
from mxnet.test_utils import set_default_context
from mxnet import npx
import time
import pickle

npx.set_np()
class Test:
    pass


op_type = 'DeepNumPy CPU'
trails = 1

if op_type == 'Official Numpy':
    import numpy as np
    import numpy_ml as ml
elif op_type == 'DeepNumPy CPU':
    from mxnet import numpy as np
    import deepnumpy_ml as ml
    set_default_context(mx.cpu(0))
elif op_type == 'DeepNumPy GPU':
    import deepnumpy_ml as ml
    from mxnet import numpy as np
    set_default_context(mx.gpu(0))

states = np.array([0, 1, 2])
n_states = len(states)
observations = np.array([0, 1])
n_observations = len(observations)
start_probability = np.array([0.2, 0.4, 0.4])
transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

# test Hidden Markov models
def test_hmm(op_type, trails):
    # generate data 
    hmm1 = ml.hmm.MultinomialHMM(transition_probability, emission_probability, start_probability)
    _states, _emissions = hmm1.generate(n_steps=10000, latent_state_types=states, obs_types=observations)
    _emissions = _emissions.reshape(10000,1)
    time_start= time.time()
    for _ in range(trails):
        hmm2 = ml.hmm.MultinomialHMM() # A=None, B=None, pi=None, eps=None
        hmm2.fit(_emissions, states, observations) # max_iter=100, tol=0.001, verbose=False
    if op_type == 'DeepNumPy CPU':
        mx.nd.waitall()
    time_end = time.time()
    print(trails, "trails:", op_type, "in dataset of shape", _emissions.shape, "consumed: ", time_end - time_start, " seconds")

# test_gp(op_type, X, trails)
# test_kr(op_type, X, trails)
# test_gmm(op_type, X, trails)
test_hmm(op_type, trails)
# lp = LineProfiler()
# lp_wrapper = lp(test_kr_explicit)
# lp_wrapper(op_type, X, trails)
# lp.print_stats()


