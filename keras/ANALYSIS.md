
# use keras

## Observations

1. Dropout helps!
    * 0.4~0.5 range
	* **NEED INPUT DROPOUT**: 0.1~0.2 range?
2. The more the neuronas in layer, the stronger the NN gets.
    * 1024-1024-1024, 512-512-512, 1024-512-256
3. Adagrad works best as a optimizer so far.
4. ReLU works best as an activation.
5. L2, L1 regularization?
6. Batchnorm, PReLU?

## Best so far

'dg8/ll-2.txt'
0.44769382476806641

{'opt': 'adagrad', 'dropout_rate': [0.5, 0.5, 0.5, 0.5],
'sgd_nesterov': False, 'activation_func': 'relu', 'nb_classes': 9,
'reg': [1e-05, 1e-05], 'sgd_mom': 0.9, 'dims': 93, 'sgd_d ecay': 0.1,
'weight_ini': 'glorot_uniform', 'input_dropout': 0.2, 'layer_size':
[1024, 1024, 1024, 1024], 'batchnorm': True, 'sgd_lr': 0.1,
'max_constraint': False, 'prelu': True}
