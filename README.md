# SVDD-Enhanced Network Embeddings for Link Prediction

This repository contains the implementation used for the additional experiments reported in the revised manuscript:

"Boosting Link Prediction with SVDD-Enhanced Network Embeddings: A Robust Framework for Complex and Attributed Networks"

The code includes experiments on Cora and CiteSeer using GNN-based baselines and the SVDD+ANE-WLDSG configuration.

## Datasets

The experiments use publicly available citation attributed networks:
- Cora
- CiteSeer

The datasets can be loaded through common graph learning libraries.

## Methods

The implemented baselines include:
- GCN
- GraphSAGE
- GAT
- GAE
- VGAE
- SVDD+ANE-WLDSG

## Metrics

The reported metrics are:
- AUC
- Average Precision (AP)

## Running the experiments

```bash
pip install -r requirements.txt
python run_gnn_baselines.py
python run_svdd_anewldsg.py
