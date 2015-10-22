#!/usr/bin/env python

from lbtoolbox.util import flipany

import os
import sys
import re
import inspect
import json
import pickle
import gzip
from os.path import join as pjoin

import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
import h5py


def here(f):
    me = inspect.getsourcefile(here)
    return pjoin(os.path.dirname(os.path.abspath(me)), f)


def imread(fname, resize=None):
    im = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("Couldn't load image " + fname)

    if resize is not None and im.shape[:2] != resize:
        im = cv2.resize(im, resize, interpolation=cv2.INTER_LANCZOS4)

    # In OpenCV, color dimension is last, but theano likes it to be first.
    # (That's map of triplets vs three maps philosophy.)
    # Also convert BGR to RGB while we're at it. Not that it makes any difference.
    im = np.rollaxis(im[:,:,::-1], 2, 0)
    return im.astype(np.float32) / 256


def scale_all(images, size=(50, 50)):
    return [cv2.resize(im, size, interpolation=cv2.INTER_LANCZOS4) for im in images]


def loadall(datadir, data):
    return zip(*[[imread(pjoin(datadir, name)), lbl, name] for lbl, files in data.items() for name in files])


def load_tosato_clf(datadir, datafile):
    data = json.load(open(pjoin(datadir, datafile)))

    tr_imgs, tr_lbls, tr_names = loadall(datadir, data['train'])
    te_imgs, te_lbls, te_names = loadall(datadir, data['test'])

    le = LabelEncoder().fit(tr_lbls)
    return (
        np.array(tr_imgs), le.transform(tr_lbls).astype(np.int32), tr_names,
        np.array(te_imgs), le.transform(te_lbls).astype(np.int32), te_names,
        le
    )


def matlab_array(mat, ref, dtype):
    N = len(ref)
    arr = np.empty(N, dtype=dtype)
    for i in range(N):
        arr[i] = mat[ref[i,0]][0,0]
    return arr


def matlab_string(obj):
    return ''.join(chr(c) for c in obj[:,0])


def matlab_strings(mat, ref):
    return [matlab_string(mat[r]) for r in ref[:,0]]


def load_tosato_idiap(datadir, datafile):
    mat_full = h5py.File(pjoin(datadir, datafile))

    def load(traintest):
        container = mat_full['or_label_' + traintest]
        pan  = matlab_array(mat_full, container['pan'],  np.float32)
        tilt = matlab_array(mat_full, container['tilt'], np.float32)
        roll = matlab_array(mat_full, container['roll'], np.float32)
        names = matlab_strings(mat_full, container['name'])
        X = np.array([imread(pjoin(datadir, traintest, name)) for name in names])
        return X, pan, tilt, roll, names

    return load('train'), load('test')


def matlab_vector(mat, col, dtype):
    N = len(mat)
    vec = np.empty(N, dtype=dtype)
    for i in range(N):
        vec[i] = mat[i][col][0,0]
    return vec


def matlab_strings2(mat, col):
    return [m[col][0] for m in mat]


def load_tosato_caviar(datadir, datafile):
    mat = loadmat(pjoin(datadir, datafile))

    def load(traintest):
        gazes = matlab_vector(mat['or_label_' + traintest][0], 0, np.float32)
        xcs = matlab_vector(mat['or_label_' + traintest][0], 1, np.float32)
        ycs = matlab_vector(mat['or_label_' + traintest][0], 2, np.float32)
        sizes = matlab_vector(mat['or_label_' + traintest][0], 3, np.float32)
        names = matlab_strings2(mat['or_label_' + traintest][0], 4)
        X = np.array([imread(pjoin(datadir, traintest, name + '.jpg')) for name in names])
        return X, gazes, xcs, ycs, sizes, names

    return load('train'), load('test')


def load_towncentre(datadir, normalize_angles=True):
    panre = re.compile('pan = ([+-]?\d+\.\d+)\n')
    valre = re.compile('valid = ([01])\n')
    angles = []
    images = []
    names = []
    for father in os.listdir(datadir):
      try:
        for son in os.listdir(pjoin(datadir, father)):
            if not son.endswith('.txt'):
                continue

            lpan, lval = open(pjoin(datadir, father, son)).readlines()
            if int(valre.match(lval).group(1)) == 0:
                continue

            angles.append(float(panre.match(lpan).group(1)))
            # Now search for the corresponding filename, unfortunately, it has more numbers encoded...
            fnames = [f for f in os.listdir(pjoin(datadir, father)) if f.startswith(son.split('.')[0]) and not f.endswith('.txt')]
            assert len(fnames) == 1, "lolwut"
            names.append(fnames[0])
            images.append(cv2.imread(pjoin(datadir, father, fnames[0]), flags=cv2.IMREAD_COLOR))
      except NotADirectoryError:
        pass

    if normalize_angles:
        angles = [(a + 360*2) % 360 for a in angles]

    return images, angles, names


def flipped_classes(X, y, n, le, old, new):
    """
    Horizontally flips all images in `X` which are labeled as `old` and label them as `new`.
    Returns the flipped X, y, n.
    """
    indices = np.where(y == le.transform(old))[0]
    return (
        flipany(X[indices], dim=3),
        np.full(len(indices), le.transform(new), dtype=y.dtype),
        tuple(n[i] for i in indices)
    )


def flipall_classes(X, y, n, le, flips):
    """
    Applies all `flips` to the whole dataset X, y, n and returns the augmented dataset.
    """
    fx, fy, fn = [], [], []
    for old, new in flips:
        a, b, c = flipped_classes(X, y, n, le, old, new)
        fx.append(a) ; fy.append(b) ; fn.append(c)
    return np.concatenate([X] + fx), np.concatenate([y] + fy), n + sum(fn, tuple())


def flipall_images(images):
    """
    Horizontally flips all given `images`, assuming `images` to be a list of HWC tensors.
    """
    return [flipany(img, dim=1) for img in images]


def flipall_angles(angles):
    """
    Horizontally flips all angles in the `angles` array.
    """
    return [360 - a for a in angles]


if __name__ == '__main__':
    datadir = here('data')

    todos = sys.argv[1:] if len(sys.argv) > 1 else ['QMUL', 'HOCoffee', 'HOC', 'HIIT', 'IDIAP', 'CAVIAR', 'TownCentre']

    if 'QMUL' in todos:
        print("Augmenting QMUL (Without \" - Copy\")... ")
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato_clf(datadir, 'QMULPoseHeads-nocopy.json')
        Xtr, ytr, ntr = flipall_classes(Xtr, ytr, ntr, le, flips=[
            ('front', 'front'),
            ('back', 'back'),
            ('background', 'background'),
            ('left', 'right'),
            ('right', 'left'),
        ])
        pickle.dump((Xtr, Xte, ytr, yte, ntr, nte, le),
                    gzip.open(pjoin(datadir, 'QMULPoseHeads-wflip.pkl.gz'), 'wb+'))
        print(len(Xtr))

    if 'HOCoffee' in todos:
        print("Augmenting HOCoffee... ")
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato_clf(datadir, 'HOCoffee.json')
        Xtr, ytr, ntr = flipall_classes(Xtr, ytr, ntr, le, flips=[
            ('frnt', 'frnt'),
            ('rear', 'rear'),
            ('left', 'rght'),
            ('rght', 'left'),
            ('frlf', 'frrg'),
            ('frrg', 'frlf'),
        ])
        pickle.dump((Xtr, Xte, ytr, yte, ntr, nte, le),
                    gzip.open(pjoin(datadir, 'HOCoffee-wflip.pkl.gz'), 'wb+'))
        print(len(Xtr))

    if 'HOC' in todos:
        print("Augmenting HOC... ")
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato_clf(datadir, 'HOC.json')
        Xtr, ytr, ntr = flipall_classes(Xtr, ytr, ntr, le, flips=[
            ('back', 'back'),
            ('front', 'front'),
            ('left', 'right'),
            ('right', 'left'),
        ])
        pickle.dump((Xtr, Xte, ytr, yte, ntr, nte, le),
                    gzip.open(pjoin(datadir, 'HOC-wflip.pkl.gz'), 'wb+'))
        print(len(Xtr))

    if 'HIIT' in todos:
        print("Augmenting HIIT... ")
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato_clf(datadir, 'HIIT6HeadPose.json')
        Xtr, ytr, ntr = flipall_classes(Xtr, ytr, ntr, le, flips=[
            ('frnt', 'frnt'),
            ('rear', 'rear'),
            ('left', 'rght'),
            ('rght', 'left'),
            ('frlf', 'frrg'),
            ('frrg', 'frlf'),
        ])
        pickle.dump((Xtr, Xte, ytr, yte, ntr, nte, le),
                    gzip.open(pjoin(datadir, 'HIIT-wflip.pkl.gz'), 'wb+'))
        print(len(Xtr))

    if 'IDIAP' in todos:
        print("Augmenting IDIAP... (lol nope, just converting)")
        # Since this one appears to already have been flipped horizontally, there's nothing to be done.
        data = load_tosato_idiap(pjoin(datadir, 'IHDPHeadPose'), 'or_label_full.mat')
        # Can't gzip due to this [Python bug](https://bugs.python.org/issue23306).
        pickle.dump(data, open(pjoin(datadir, 'IDIAP.pkl'), 'wb+'))
        print(len(data[0][0]))

    if 'CAVIAR' in todos:
        print("Augmenting CAVIAR... (all hope is lost, no augmentation is done)")
        data = load_tosato_caviar(pjoin(datadir, 'CAVIARShoppingCenterFull'), 'or_label.mat')
        pickle.dump(data, gzip.open(pjoin(datadir, 'CAVIAR-c.pkl.gz'), 'wb+'))
        print(len(data[0][0]))
        data = load_tosato_caviar(pjoin(datadir, 'CAVIARShoppingCenterFullOccl'), 'or_label.mat')
        pickle.dump(data, gzip.open(pjoin(datadir, 'CAVIAR-o.pkl.gz'), 'wb+'))
        print(len(data[0][0]))


    if 'TownCentre' in todos:
        print("Augmenting TownCentre... ")
        bbtc_img, bbtc_a, bbtc_n = load_towncentre('data/TownCentreHeadImages')
        bbtc_img50 = scale_all(bbtc_img, (50, 50))

        Xtc = np.array(bbtc_img50 + flipall_images(bbtc_img50))
        ytc = np.array(bbtc_a + flipall_angles(bbtc_a))
        ntc = bbtc_n + bbtc_n

        # BHWC -> BCHW
        Xtc = np.rollaxis(Xtc, 3, 1)

        pickle.dump((Xtc, ytc, ntc),
                    gzip.open(pjoin(datadir, 'TownCentre.pkl.gz'), 'wb+'))
