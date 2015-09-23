# susyML
Unsupervised machine learning for SUSY signal detection

1. SOM: 
..1. SOM_theano.py: class file for a SOM object 
..2. SOM_run.py: the script used to train a SOM 
..3. SOM_test.py: the script used to test a SOM 

2. NADE: 
..1. OrderlessNADE.py: the wrap for an OrderlessNADE object. "__main__" method has code for training and testing
..2. the rest: support for OrderlessNADE obtained from the original NADE script

3. Datasets: 
..1. DatasetProcessor.py: a script that selects the data according to specified cuts
..2. FileProcessor.py: a script that does stage 1 selection, calculates the Razor variables and converts .root files into .hdf5 
..3. localtest.py: testing script for my local directory 
..4. move.py: a script that copies all the .root files from eos to my directory on lxplus

4. Razor: 
..1. Razor.py: a script that bins the razor plane according to the data and performs the likelihood test 

5. ShowData.py: A plotting script for show the distribution in each variable
