#!/usr/bin/env python

from lbtoolbox.download import download

import os
import inspect
import tarfile


def here(f):
    me = inspect.getsourcefile(here)
    return os.path.join(os.path.dirname(os.path.abspath(me)), f)


def download_extract(urlbase, name, into):
    print("Downloading " + name)
    fname = download(os.path.join(urlbase, name), into)
    print("Extracting...")
    with tarfile.open(fname) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, path=into)


if __name__ == '__main__':
    baseurl = 'https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/'
    datadir = here('data')

    # First, download the Tosato datasets.
    download_extract(baseurl, 'CAVIARShoppingCenterFullOccl.tar.bz2', into=datadir)
    download_extract(baseurl, 'CAVIARShoppingCenterFull.tar.bz2', into=datadir)
    download_extract(baseurl, 'HIIT6HeadPose.tar.bz2', into=datadir)
    download_extract(baseurl, 'HOC.tar.bz2', into=datadir)
    download_extract(baseurl, 'HOCoffee.tar.bz2', into=datadir)
    download_extract(baseurl, 'IHDPHeadPose.tar.bz2', into=datadir)
    download_extract(baseurl, 'QMULPoseHeads.tar.bz2', into=datadir)

    # Second, Benfold's TownCentre dataset.
    download_extract(baseurl, 'TownCentreHeadImages.tar.bz2', into=datadir)

    print("Done.")
