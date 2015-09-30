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

## `Inspection - Tosato Classification.ipynb`

While browsing through Tosato's classification datasets, I noticed some curious filenames, and used this notebook to investigate dataset-weirdnesses.
Note that I did not attempt to fix any of the weirdness, as it would render comparisons meaningless.

Results:

- `QMUL` contains files suffixed with ` - Copy`, such as `001633.jpg` followed by `001633 - Copy.jpg`.
  These are actual exact copies. Whether this is unintended, or a hacky way to give some data more weight, I don't know.
- `QMUL` contains files suffixed with `_f`, but they are unique and not flips of other files, as one might guess.
- `HOCoffee` has a lot of identical filenames but of different types, e.g. `head (1).png` vs. `head (1).jpg`.
- `HOC` (aka `ETHZ`) contains files with matching names such as `norm_0002002.jpg` and `norm_0002002 (3).jpg`. These don't seem related, though.

Given the above, I exhaustively searched for mirroring in the dataset but found none, meaning that mirroring is a valid data augmentation step.

Besides the above, this notebook also plots the label distributions, which is mildly interesting.

## `prepare_data.py`

Simply run this script without any arguments.
It will load all the raw data, apply horizontal flip augmentation and then save the data into gzipped pickles for easier use from notebooks.
This can take some time, even on fast machines.
It's not optimized and took ~50min on my high-end desktop, but it's a one-off thing so just read some paper while it's working.
