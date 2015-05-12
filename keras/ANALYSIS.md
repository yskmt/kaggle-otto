
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

5: 0.455

{'opt': 'adagrad', 'dropout_rate': [0.4, 0.4, 0.4], 'W_reg': <function
l2wrap at 0x7f2841edc0c8>, 'nb_classes': 9, 'b_reg': <function l2wrap
at 0x7f2841edc140>, 'activation_func': 'relu', 'layer_size':
[1024, 1024, 1024], 'sgd_mom': 0.9, 'sgd_decay': 0.1, 'weight_ini':
'glorot_uniform', 'input_dropout': 0.2, 'sgd_nesterov': False,
'batchnorm': True, 'sgd_lr': 0.1, 'max_constraint': False, 'prelu':
True}
