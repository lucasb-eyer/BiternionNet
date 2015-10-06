#!/usr/bin/env python

from lbtoolbox.util import flipany

import os
import inspect
import json
import pickle
import gzip
from os.path import join as pjoin

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


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


def loadall(datadir, data):
    return zip(*[[imread(pjoin(datadir, name)), lbl, name] for lbl, files in data.items() for name in files])


def load_tosato(datadir, datafile):
    data = json.load(open(pjoin(datadir, datafile)))

    tr_imgs, tr_lbls, tr_names = loadall(datadir, data['train'])
    te_imgs, te_lbls, te_names = loadall(datadir, data['test'])

    le = LabelEncoder().fit(tr_lbls)
    return (
        np.array(tr_imgs), le.transform(tr_lbls).astype(np.int32), tr_names,
        np.array(te_imgs), le.transform(te_lbls).astype(np.int32), te_names,
        le
    )


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


if __name__ == '__main__':
    datadir = here('data')

    todos = sys.argv[1:] if len(sys.argv) > 1 else ['QMUL', 'HOCoffee', 'HOC', 'HIIT', 'TownCentre']

    if 'QMUL' in todos:
        print("Augmenting QMUL (Without \" - Copy\")... ")
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato(datadir, 'QMULPoseHeads-nocopy.json')
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
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato(datadir, 'HOCoffee.json')
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
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato(datadir, 'HOC.json')
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
        Xtr, ytr, ntr, Xte, yte, nte, le = load_tosato(datadir, 'HIIT6HeadPose.json')
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
