# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def get_params():
    paramname = {'diversify': [' --latent_domain_num ',
                               ' --alpha1 ',  ' --alpha ', ' --lam ']}
    paramlist = {
        'diversify': [[2, 3, 5, 10, 20], [0.1, 0.5, 1], [0.1, 1, 10], [0], [[1, 150], [3, 50], [5, 30], [10, 15], [30, 5]]]
    }
    return paramname, paramlist

# Add this at the bottom of your existing params.py
gnn_params = {
    "gcn_num_layers": 2,
    "gcn_hidden_dim": 64,
    "lstm_hidden": 64,
    "feature_output_dim": 128,
    "gnn_dropout": 0.1,
}
