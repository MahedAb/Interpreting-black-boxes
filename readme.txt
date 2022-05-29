This file provides information about running the code to reproduce the results mentioned in the paper. Details of the sub-directories are mentioned below

- docs: contains the requirements.txt file needed to install all the libraries/packages required to execute the code.

- experiment1, experiment2, experiment3: contain the source code and other dependencies required to run those experiments. 


============================================================================
Experiment 1
============================================================================

To execute experiment 1, i.e., the experiment where we model bi-variate functions with known functional form, please execute the command below:

python3 experiment1/wrapper_exp1.py --M 20 --s 4 --max_iter 30 --k 20 --lr 0.1 > experiment1/log_exp1.txt 


The command above will use our proposed SMPF approach to learn symbolic metamodels for each of the four functions. After successful completion of the simulation, the generated log file in experiment1 folder contains the performance of the method for each function and the expression of the learned metamodels.


============================================================================
Experiment 2 (Instance-wise Feature importance in Appendix)
============================================================================

The experiment 2 involves using our proposed to generate instance-wise feature importance scores for explaining the predictions of a neural network model. We compare the performance of our approach with seven other methods - six of which are taken from the paper that introduced the SM approach, and the seventh is the SP approach. To run the experiment 2, please execute the command below

python3 experiment2/wrapper_exp2.py --M 20 --s 2 --max_iter 10 --k 10 --lr 0.01 --out_dir experiment2/

Successful completion of the above command will create a pickle file called 'median_smgp.pickle' in the path 'experiment2'. This file contains the median ranks obtained using our approach for each dataset. Importantly, the path 'experiment2' also contains a pickle file called 'median_others.pickle' which contains the results for all the seven methods we compare our method against.

To plot the figure 3 in the paper that compares the different methods, please execute the command below

python3 experiment2/plot_fig_exp2.py

On completion the above command will create the figure called 'fig_exp2.pdf' in the path 'experiment2'.


============================================================================
Experiment 3
============================================================================

In order to run experiment 3, please use the bash script file named - "run_exp3.sh" in the "experiment3" folder. The script file on execution using the command below trains metamodels (using SMPF) for two black-box models and the 'yacht' dataset from the UCI repository. The command to execute the bash file and store results in a log file is 

experiment3/run_exp3.sh > experiment3/log_exp3.txt
