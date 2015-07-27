# BiternionNet
Code used to produce the results of the paper "BiternionNets: Continuous Head Pose Regression from Discrete Training Labels"

Note that I am still in the process of cleaning up my code so that it can (hopefully) be used by anyone besides myself,
thus not everything is in here yet. It's on its way.

# Requirements

For running the experiments, you'll need to install recent versions of at least the following:

- Python3
- NumPy
- SciPy
- Matplotlib
- OpenCV with Python bindings
- Theano
- Jupyter notebook (aka IPython)

I've written a [tutorial on installing all of them](http://lb.eyer.be/a/sci-env.html).
You'll also need to clone and install my own toolbox(es):

- DeepFried: `pip install git+https://github.com/lucasb-eyer/DeepFried.git`
- lbtoolbox: `pip install git+https://github.com/lucasb-eyer/lbtoolbox.git`

Since most of my experiments are written in Jupyter notebook, you can directly look at them on Github or start a local server:

```
cd THIS_REPO
ipython notebook
```

# What's what?

## `download_data.py`

Simply running this file will download and extract all benchmark datasets necessary for the comparisons done in the paper.
You *need* to run this file before most of the notebooks will work.
