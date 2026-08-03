from setuptools import find_packages, setup

setup(
    name='stflow',
    packages=['stflow'],
    install_requires=[
        'einops',
        'h5py',
        'scanpy',
        'torchmetrics',
    ],
)
