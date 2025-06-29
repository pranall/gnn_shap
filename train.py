# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
import torch
import torch.nn as nn

# === GNN imports ===
from models.gnn_extractor import TemporalGCN, build_correlation_graph
from diversify.utils.params import gnn_params

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    best_valid_acc, target_acc = 0, 0

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()

    # ===== GNN feature extractor integration =====
    use_gnn = getattr(args, "use_gnn", 0)
    gnn = None
    if use_gnn:
        # Assumes data shape is [batch, channels, timesteps]
        example_batch = next(iter(train_loader))[0] if hasattr(train_loader, '__iter__') else None
        in_channels = example_batch.shape[1] if example_batch is not None else 8
        gnn = TemporalGCN(
            in_channels=in_channels,
            hidden_dim=gnn_params["gcn_hidden_dim"],
            num_layers=gnn_params["gcn_num_layers"],
            lstm_hidden=gnn_params["lstm_hidden"],
            output_dim=gnn_params["feature_output_dim"]
        ).cuda()
        # >>>>> KEY CHANGE: Overwrite featurizer with identity for GNN <<<<<
        algorithm.featurizer = nn.Identity()
        print('[INFO] GNN feature extractor initialized. CNN featurizer is bypassed.')
        # >>>>> NEW: Patch bottleneck(s) for GNN feature size <<<<<
        gnn_out_dim = gnn.out.out_features
        if hasattr(algorithm, "bottleneck"):
            algorithm.bottleneck = nn.Linear(gnn_out_dim, 256).cuda()
            print(f"[INFO] Bottleneck adjusted for GNN: {gnn_out_dim} -> 256")
        if hasattr(algorithm, "abottleneck"):
            algorithm.abottleneck = nn.Linear(gnn_out_dim, 256).cuda()
            print(f"[INFO] Adversarial bottleneck adjusted for GNN: {gnn_out_dim} -> 256")
        if hasattr(algorithm, "dbottleneck"):
            algorithm.dbottleneck = nn.Linear(gnn_out_dim, 256).cuda()
            print(f"[INFO] Domain bottleneck (dbottleneck) adjusted for GNN: {gnn_out_dim} -> 256")
        # === NEW for GNN in set_dlabel (do NOT remove these lines) ===
        algorithm.gnn_extractor = gnn
        algorithm.use_gnn = True

    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                # === GNN: extract features if enabled ===
                if use_gnn and gnn is not None:
                    batch_x = data[0] if isinstance(data, (list, tuple)) else data
                    if len(batch_x.shape) == 4 and batch_x.shape[2] == 1:
                        batch_x = batch_x.squeeze(2)
                    gnn_graphs = build_correlation_graph(batch_x.cuda())
                    from torch_geometric.loader import DataLoader as GeoDataLoader
                    geo_loader = GeoDataLoader(gnn_graphs, batch_size=len(gnn_graphs))
                    for graph_batch in geo_loader:
                        graph_batch = graph_batch.cuda()
                        gnn_features = gnn(graph_batch)
                    # >>>>> Only pass GNN features and label(s) forward <<<<<
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        data = (gnn_features, *data[1:])
                    else:
                        data = gnn_features
                # === END GNN block ===

                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                # === GNN: extract features if enabled ===
                if use_gnn and gnn is not None:
                    batch_x = data[0] if isinstance(data, (list, tuple)) else data
                    if len(batch_x.shape) == 4 and batch_x.shape[2] == 1:
                        batch_x = batch_x.squeeze(2)
                    gnn_graphs = build_correlation_graph(batch_x.cuda())
                    from torch_geometric.loader import DataLoader as GeoDataLoader
                    geo_loader = GeoDataLoader(gnn_graphs, batch_size=len(gnn_graphs))
                    for graph_batch in geo_loader:
                        graph_batch = graph_batch.cuda()
                        gnn_features = gnn(graph_batch)
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        data = (gnn_features, *data[1:])
                    else:
                        data = gnn_features
                # === END GNN block ===

                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('====Domain-invariant feature learning====')

        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                # === GNN: extract features if enabled ===
                if use_gnn and gnn is not None:
                    batch_x = data[0] if isinstance(data, (list, tuple)) else data
                    if len(batch_x.shape) == 4 and batch_x.shape[2] == 1:
                        batch_x = batch_x.squeeze(2)
                    gnn_graphs = build_correlation_graph(batch_x.cuda())
                    from torch_geometric.loader import DataLoader as GeoDataLoader
                    geo_loader = GeoDataLoader(gnn_graphs, batch_size=len(gnn_graphs))
                    for graph_batch in geo_loader:
                        graph_batch = graph_batch.cuda()
                        gnn_features = gnn(graph_batch)
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        data = (gnn_features, *data[1:])
                    else:
                        data = gnn_features
                # === END GNN block ===

                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': step,
            }

            results['train_acc'] = modelopera.accuracy(
                algorithm, train_loader_noshuffle, None)

            acc = modelopera.accuracy(algorithm, valid_loader, None)
            results['valid_acc'] = acc

            acc = modelopera.accuracy(algorithm, target_loader, None)
            results['target_acc'] = acc

            for key in loss_list:
                results[key+'_loss'] = step_vals[key]
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
