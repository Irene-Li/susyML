# susyML
Unsupervised machine learning for SUSY signal detection

1. SOM: 
  * SOM_theano.py: class file for a SOM object 
  * SOM_run.py: the script used to train a SOM 
  * SOM_test.py: the script used to test a SOM 

2. NADE: 
  * OrderlessNADE.py: the wrap for an OrderlessNADE object. "__main__" method has code for training and testing
  * the rest: support for OrderlessNADE obtained from the original NADE script

3. Datasets: 
  * DatasetProcessor.py: a script that selects the data according to specified cuts
  * FileProcessor.py: a script that does stage 1 selection, calculates the Razor variables and converts .root files into .hdf5 
  * localtest.py: testing script for my local directory 
  * move.py: a script that copies all the .root files from eos to my directory on lxplus

4. Razor: 
  * Razor.py: a script that bins the razor plane according to the data and performs the likelihood test 

5. ShowData.py: A plotting script for show the distribution in each variable
