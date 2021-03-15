This repository has been bootstrapped from the codebase available (accessed on 13-03-21) on the paper [Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/pdf/1803.09587.pdf).

We developed extension of the S-KNN model that allows pre-filtering and post-filtering
strategies based on the timestamp of the interactions. We implemented the following algorithms:

- Session-KNN (in wknn): We created a class that extended the original S-KNN model that is able to use contextual extensions, use different similarity functions, sampling strategies and scoring functions.
- Context-IKNN: Item-based KNN that uses only information of interactions that happened in the same context.
- Item2Vec: An adaptation of the Word2Vec model that creates item embeddings by analyzing sessions as sentences and words as items.
- Session2Vec: An algorithm that creates item embeddings similarly to the Item2Vec algorithm, but combines item embeddings in order to create session embeddings.

We also adapted the `run_test.py` script so that you can run the tests based on the dataset.

Below you can find the README made available from the paper's authors.


Requirements
----------------------

* Anaconda 4.X (Python 3.5+)
* Pympler
* NumPy
* SciPy
* BLAS
* Pandas
* Theano
* Tensorflow
* Suitable datasets (https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0)

Installation (tested on Debian 8)
----------------------
1. Download and install Anaconda (https://www.continuum.io/).
	* NumPy, SciPy, BLAS, Pandas should automatically be included.
2. Install build essentials
	* apt-get install build-essential
	* On Windows the installation of mingw with Anaconda should work.
		* conda install mingw
3. Install Theano, Tensorflow and additional packages
	* conda install theano tensorflow pympler

Usage
----------------------
### Preprocessing
1. Unzip any dataset file to the data folder, i.e., rsc15-clicks.dat will then be in the folder data/rsc15/raw
2. Open the script run_preprocessing*.py to configure the preprocessing method and parameters
	* run_preprocessing_rsc15.py is for the RecSys challenge dataset.
	* run_preprocessing_tmall.py is for the TMall logs.
	* run_preprocessing_retailrocket.py is for the Retailrocket competition dataset.
	* run_preprocessing_clef.py is for the Plista challenge dataset.
	* run_preprocessing_music.py is for all music datasets (configuration of the input and output path inside the file).
3. Run the script

### Running experiments
1. You must have run the preprocessing scripts previously
2. Open and edit one of the run_test*.py scripts
	* run_test.py evaluates predictions for single split in terms of just the next item (HR@X and MRR@X)
	* run_test_pr.py evaluates predictions for single split in terms of all remaining items in the session (P@X, R@X, and MAP@X)
	* run_test_window.py evaluates predictions for sliding window split in terms of the next item (HR@X and MRR@X)
 	* run_test_buys.py evaluates buy events in the sessions (only for the rsc15 dataset).The script run_preprocessing.py must have been executed with method "buys" before.
	* The usage of all algorithms is exemplarily shown in the script.
3. Run the script
4. Results and run times will be displayed and saved to the results folder as configured
