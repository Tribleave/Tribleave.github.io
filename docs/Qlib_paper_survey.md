# Survey of Qlib

## Quant Model Zoo

### GBDT-based models

- XGBoost, 2016 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/xgboost.py)] [[paper](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml)]
- LightGBM, 2017 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/gbdt.py)] [[paper](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)]
- CatBoost, 2018 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/catboost_model.py)] [[paper](https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/CatBoost/workflow_config_catboost_Alpha158.yaml)]
- Localformer [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_localformer_ts.py)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/Localformer/workflow_config_localformer_Alpha158.yaml)]

### RNN-based models

#### baselines

- LSTM, 1997 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_lstm_ts.py)] [[paper](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml)]
- GRU, 2014 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_gru_ts.py)] [[paper](https://aclanthology.org/D14-1179.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml)]
- Transformer, 2017 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_transformer_ts.py)] [[paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/Transformer/workflow_config_transformer_Alpha158.yaml)]

#### Time series prediction

- ALSTM, 2017 (A dual-stage attention-based recurrent neural network for time series prediction) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_alstm_ts.py)] [[paper](https://www.ijcai.org/Proceedings/2017/0366.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/ALSTM/workflow_config_alstm_Alpha158.yaml)]
- SFM, 2017 (State-Frequency-Memory) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_sfm.py)] [[paper](http://www.eecs.ucf.edu/~gqi/publications/kdd2017_stock.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/SFM/workflow_config_sfm_Alpha360.yaml)]
- TFT, 2019 (Temporal Fusion Transformers) [[code](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TFT/tft.py)] [[paper](https://arxiv.org/pdf/1912.09363.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TFT/workflow_config_tft_Alpha158.yaml)] [[pytorch implmentation](https://github.com/mattsherar/Temporal_Fusion_Transform)]
- TabNet, 2019 (TabNet: Attentive Interpretable Tabular Learning) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tabnet.py)] [[paper](https://arxiv.org/pdf/1908.07442.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TabNet/workflow_config_TabNet_Alpha158.yaml)]
- TRA, 2021 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tra.py)] [[paper](http://arxiv.org/abs/2106.12950)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TRA/workflow_config_tra_Alpha158.yaml)]
- ADARNN, 2021 (AdaRNN: Adaptive Learning and Forecasting for Time Series) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_adarnn.py)] [[paper](https://arxiv.org/pdf/2108.04443.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/ADARNN/workflow_config_adarnn_Alpha360.yaml)]
- ADD, 2020 (ADD: Augmented Disentanglement Distillation Framework for Improving Stock Trend Forecasting) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_add.py)] [[paper](https://arxiv.org/abs/2012.06289)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/ADD/workflow_config_add_Alpha360.yaml)]

## GNN-based Models

- GAT, 2017 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_gats_ts.py)] [[paper](https://arxiv.org/pdf/1710.10903.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml)]

## Ensemble Models

- DoubleEnsemble, 2020 [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/double_ensemble.py)] [[paper](https://arxiv.org/pdf/2010.01265.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/DoubleEnsemble/workflow_config_doubleensemble_Alpha158.yaml)]

## MLP-based Models

- MLP [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/MLP/workflow_config_mlp_Alpha158.yaml)]
- TCTS, 2021 (Temporally Correlated Task Scheduling for Sequence Learning) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcts.py)] [[paper](http://proceedings.mlr.press/v139/wu21e/wu21e.pdf)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TCTS/workflow_config_tcts_Alpha360.yaml)]


## CNN-based models

- TCN, 2018 (An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling) [[code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcn_ts.py)] [[paper](https://arxiv.org/abs/1803.01271)] [[config](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TCN/workflow_config_tcn_Alpha158.yaml)]