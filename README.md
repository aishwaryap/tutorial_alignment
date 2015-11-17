# tutorial_alignment
This project uses latent models to align different tutorials for the same task. The model should also discriminate between tutorials of different tasks.

HMM code has been obtained from https://github.com/guyz/HMM

HMM
===

A numpy/python-only Hidden Markov Models framework. No other dependencies are required.

This implementation (like many others) is based on the paper:
"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, LR RABINER 1989"

Major supported features:

* Discrete HMMs
* Continuous HMMs - Gaussian Mixtures
* Supports a variable number of features
* Easily extendable with other types of probablistic models (simply override the PDF. Refer to 'GMHMM.py' for more information)
* Non-linear weighing functions - can be useful when working with a time-series

Open concerns:
* Examples are somewhat out-dated
* Convergence isn't guaranteed when using certain weighing functions 
