# Bayesian ERFNet - Pytorch Lightning Implementation

This repository is a submodule of our paper "Informative Path Planning for Active Learning in 
Aerial Semantic Mapping". The repository provides Pytorch Lightning implementations to train and
evaluate our proposed Bayesian ERFNet for per-pixel model uncertainty quantification with Monte-Carlo 
dropout. The paper can be found [here](https://arxiv.org/pdf/2203.01652.pdf). If you found this work useful for your own research, feel free to cite it.

```commandline
@inproceedings{ruckin2022informative,
  title={Informative Path Planning for Active Learning in Aerial Semantic Mapping},
  author={R{\"u}ckin, Julius and Jin, Liren and Magistri, Federico and Stachniss, Cyrill and Popovi{\'c}, Marija},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022},
  organization={IEEE}
}
```

## Installation & Setup

```bash
pip3 install -r requirements.txt
```

### Docker

Requires [docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/).

First, build the pipeline:
```bash
docker-compose build
```

To start the training pipeline and tensorboard:
```bash
docker-compose up
```

## Development

### Style Guidelines

In general, we follow the Python [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines. Please install [black](https://pypi.org/project/black/) to format your python code properly.
To run the black code formatter, use the following command:

```commandline
black -l 120 path/to/python/module/or/package/
```

To optimize and clean up your imports, feel free to have a look at this solution for [PyCharm](https://www.jetbrains.com/pycharm/guide/tips/optimize-imports/).

### Maintainer

Julius Rückin, [jrueckin@uni-bonn.de](mailto:jrueckin@uni-bonn.de), Ph.D. student at [PhenoRob - University of Bonn](https://www.phenorob.de/)

## Acknowledgement

We would like to thank Jan Weyler for providing a PyTorch Lightning implementation of ERFNet.
Our Bayesian-ERFNet implementation builds upon Jan's ERFNet implementation. 

## Funding

This work was funded by the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) under Germany’s Excellence Strategy - EXC
2070 – 390732324. Authors are with the Cluster of Excellence PhenoRob,
Institute of Geodesy and Geoinformation, University of Bonn.