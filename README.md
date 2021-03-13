# Differentiation of Blackbox Combinatorial Solvers

This repository provides code for the paper [Differentiation of Blackbox Combinatorial Solvers](http://arxiv.org/abs/1912.02175).

By Marin Vlastelica*, Anselm Paulus*, Vít Musil, [Georg Martius](http://georg.playfulmachines.com/) and [Michal Rolínek](https://scholar.google.de/citations?user=DVdSTFQAAAAJ&hl=en).

[Autonomous Learning Group](https://al.is.tuebingen.mpg.de/), [Max Planck Institute for Intelligent Systems](https://is.tuebingen.mpg.de/).

For a condensed version containing only the modules (along with additional solvers) see [this repository](https://github.com/martius-lab/blackbox-backprop).

## Table of Contents
0. [Introduction](#introduction)
1. [Installation](#installation)
2. [Usage](#usage)
3. [Notes](#notes)



## Introduction

This repository provides a visualization for all the datasets used in
[Differentiation of Blackbox Combinatorial Solvers](http://arxiv.org/abs/1912.02175).
Additionally, the training code for the Warcraft Shortest Path experiment is provided.

*Disclaimer*: This code is a PROTOTYPE. It should work fine but use at your own risk.

## Installation

First install the requirements with via one of the following options:

- Option 1 (requires pipenv):

  ``pipenv install`` (use --skip-lock flag for speedup at your own risk)
      
  ``pipenv shell``

  If the installation causes problems, the following could be a fix:
  ``sudo apt install python3.6-dev``

- Option 2 (requires python 3.6):

  ``pip3 install -r requirements.txt``
  

Next, download our [datasets](https://edmond.mpdl.mpg.de/imeji/collection/tGU9ok0_m2CVfHI8?q=) and extract each dataset to the data/ directory in this repository.


## Usage

- Dataset visualization:

    For an easy overview over the datasets use the data/data_visualization.ipynb jupyter notebook.

- Warcraft shortest path experiment: 
    
    Change directory to project root folder.
    
    Run Warcraft shortest path experiment with gradients through Dijkstra:
    
    ``python main.py settings/warcraft_shortest_path/12x12_combresnet.json``
    
    Run Warcraft shortest path baseline ResNet18 experiment:
    
    ``python main.py settings/warcraft_shortest_path/12x12_baseline.json``
    
    The results are stored in the results directory in the project root folder.


## Notes

*Contribute*: If you spot a bug or some incompatibility, raise an issue or contribute via a pull request! Thank you!
