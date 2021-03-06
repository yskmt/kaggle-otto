 
# XGB best result

0.444807
xgb3.e74518 
{'eval_metric': 'mlogloss', 'early_stopping_rounds': 10,
'colsample_bytree': '0.5', 'num_class': 9, 'silent': 1, 'nthread': 16,
'min_child_weight': '4', 'subsample': '0.8', 'eta': '0.0125',
'objective': 'multi:softprob', 'max_depth': '14', 'gamma': '0.025'}



## confusion matrix by xgb
[[ 206   10    4    0    1   27   21   56   80]
 [   3 2759  369   31    4    6   23    7    2]
 [   0  686  841   23    0    3   29    4    4]
 [   0  149   76  273    2   21    9    2    0]
 [   0   12    3    0  550    0    0    1    1]
 [  10   19    5    5    3 2623   32   31   35]
 [   9   53   43    6    1   40  398   18    2]
 [  17   14    6    0    1   35    9 1639   26]
 [  17   16    1    0    1   30    7   34  891]]

# RGF best result

Expo L2: 0.05, sL2: 0.005
expo2_0.0050000.05
[[ 0.53798638  0.00422792]
 [ 0.50277415  0.00487947]
 [ 0.48711101  0.00495653]
 [ 0.47855525  0.00514518]
 [ 0.47362229  0.0054265 ]
 [ 0.47027597  0.00558858]
 [ 0.46832914  0.00590766]
 [ 0.46723642  0.0058981 ]]

## confusion matrix by rgf
[[ 197   11    4    0    4   28   21   56   84]
 [   2 2709  404   38    6    5   29    6    5]
 [   0  666  852   34    0    3   29    3    3]
 [   0  151   75  274    2   23    5    2    0]
 [   0   12    3    0  549    0    1    1    1]
 [   8   19    2    4    2 2615   39   36   38]
 [   8   55   41    6    5   39  395   18    3]
 [  21   14    7    0    2   36   12 1624   31]
 [  13   16    0    0    2   35    4   32  895]]

# SVC best result
{"kernel": "rbf", "C": 20.0, "simdir": "sim_opt_2", "gamma": 0.0}
[0.53504065754888119 0.0062319786914490404]


## confusion matrix by svc
[[ 194   13    2    0    1   20   22   77   76]
 [   1 2810  308   37    3    8   23    9    5]
 [   0  858  660   29    0    5   31    5    2]
 [   0  205   62  233    3   18    9    1    1]
 [   2   17    1    0  545    1    0    1    0]
 [  13   16    2    7    1 2593   29   65   37]
 [  11   60   30    7    3   47  369   37    6]
 [  20   13    9    0    1   40   15 1617   32]
 [  24   14    1    0    0   35    7   40  876]]


# NN best result
[0.48618308965290896 0.027018410968709172]

## confusion matrix by nn
[[ 247    8    3    1    0   30   11   45   48]
 [   0 2928  281   52    2    6   27    9    5]
 [   2  851  687   16    0    0   38    5    0]
 [   1  166   63  280    2   15   15    2    0]
 [   0    9    1    0  499    3    2    1    0]
 [  12   24    5    6    0 2678   36   35   22]
 [   7   52   22    0    4   31  405   19    1]
 [  22   17    2    0    1   33   13 1538   21]
 [  53   14    0    1    3   44    6   34  853]]


# Keras best results 
http://keras.io/optimizers/

0.45346915721893311
{'opt': 'adagrad', 'dropout_rate': [0.4, 0.4, 0.4], 'sgd_nesterov':
False, 'activation_func': 'relu', 'nb_classes': 9, 'reg':
[1e-05, 1e-05], 'sgd_mom': 0.9, 'dims': 93, 'sgd_deay': 0.1,
'weight_ini': 'glorot_uniform', 'input_dropout': 0.2, 'layer_size':
[1024, 1024, 1024], 'batchnorm': True, 'sgd_lr': 0.1,
'max_constraint': False, 'prelu': True}


# Model mixing

Try:

1. Linear model addition with optimized model contributions:
   SVC+XGB+NN+(RGF?)
2. Optimize the model contributions class-wise: some models are
   stronger at classfying certain classes.
