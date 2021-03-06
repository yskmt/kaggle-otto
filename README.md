# Kaggle's Otto Classification Challenge


[Challenge website](https://www.kaggle.com/c/otto-group-product-classification-challenge)


## Regularized Greedy Forest


## Gradient Boosting Classifier

* Use [xgboost](https://github.com/dmlc/xgboost)
* Cross validation: 
  - Tuning parameters:
    - learning rate (eta)
    - subsampling rate (subsample)
    - max depth of the individual trees (max_depth)
    - number of rounds: 4000
  - Observations:
    - Tree depth = 10 works so much better than ~3
  - Best so far:
    - eta=0.05, subsample=0.75, num_rounds=4000, max_depth=2
      - cv test logloss error = 0.536728+0.017996


## Neural net

### lasagne + nolearn

* [quick starting guide](http://nbviewer.ipython.org/github/ottogroup/kaggle/blob/master/Otto_Group_Competition.ipynb)

#### Note on use of THEANO on euclid


    module load cuda
    export CUDA_ROOT=/opt/apps/sysnet/cuda/6.0/cudatk
    THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'  python <myscript>.py

More on [here](http://www.deeplearning.net/software/theano/library/config.html)

### keras


## Random Forest
* Some sampleing

## Adaboost

* Not very good.


## Useful resources

* Kaggle forum
  * [Neural net](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13016/neural-nets-in-sklearn/68544#post68544)
  * [Single model results](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13302/how-far-could-you-get-with-just-one-model)

    
