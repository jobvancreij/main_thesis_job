# Main codes Thesis Job

The goal of this repo is initialzing/testing/evaluating the experiments for the thesis
"Algorithmic trading of Ethereum/Bitcoin coinpair with deep learning and big data". <br> 

This means that this repo has several goals: 
* Initialize the experiments, which means adding the settings to firestore
* Running the experiments (if not ran on colab)
* Training the optimized models 
* Running the trading algorithms 
* Visualization
* Wilcoxen test

To run install depedencies acces is needed to: 
* https://github.com/jobvancreij/LJT_database (private)
* https://github.com/jobvancreij/LJT_helper_functions (private)
* https://github.com/jobvancreij/LJT_pubsub_codes (private)
* https://github.com/jobvancreij/deep_learning_models
* https://github.com/jobvancreij/trading_algorithms
    
    
Before running any of the code the installs should be done. This can be done by running: 
```console
python install_dependencies.py 
```

This code installs the custom libraries as well as the other pip installs.



### 1. Initialize the experiments 
Before any experiment can be done the experiments have to be initialized. You can choose to initialize
for coinpairs (ETHBTC or BNBBTC) and algorithms (LSTM or GRU). Other settings such as trainsize etc 
can be changed in the init_experiment_general.py  You have to type in "jaikwil" as conformation 
that you want to start the code
```console
python initialize_experiments/init_experiments_general.py {coinpair} {deep learning model} 
```

The init_experiment_1 function calculates which window sizes are most interesting and uploads the 
settings for the experiments to firestore. Init_experiment_2 uploads the settings for the hyperopt
algorithm to firestore. They are started in a similar way as general initializer. 

### 2 Run experiments 

Experiment 1 searches for the best window and forecast horizon. It can be initalized by: 

```console
python expierments/experiment_1.py {coinpair} {deep learning model} 
```

Experiment 2 runs the hyperopt algorithm. It is currently only build for Linux 
computers/servers. It runs with  
```console
sudo python3 expierments/experiment_2.py ETHBTC LSTM & sleep 480 && hyperopt-mongo-worker --mongo=34.66.233.70:5000/eval_db --poll-interval=0.1 
```

### 3 Notebooks 
* Evaluate hyperopt algorithm: can be used to check the scores when hyperopt is running. 
* Train final models: Code to train the best found model. Since optimizers does not return the model 
itself
* Trading strategy implemented: Run the trading strategies / baseline models 
* Wilcoxon test: run the Wilcoxen test on the results of trading algorithm
* Visualize trading strategies: make plots of the trading strategies 
* Train model extended dataset: Run the models on the increases dataset, which is only used in the 
discussion of the paper 




### Extra:  Install cuda windows (tensorflow==2.0.0)
* Step 1: Installal Cuda dev | 
Version --> https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10
* Step 2: download folder <br>
* Step 3: Copy paste the files in the folder to the exact map in the cuda folder
* Step 4: download --> https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-windows10-x64-v7.6.5.32.zip
* step 5:add this to path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin 
