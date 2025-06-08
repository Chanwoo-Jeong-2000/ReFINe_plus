# [Under Review] Modeling of Negative Feedback Refined via LLM in Recommender Systems

# Requirements
python 3.9.20, cuda 11.8, and the following installations:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install pandas
pip install scikit-learn
pip install transformers
pip install ptyprocess ipykernel pyzmq -U --force-reinstall
```

# Run
##### ML-100K
```
python main.py --dataset ML-100K
```
##### ML-1M
```
python main.py --dataset ML-1M
```
##### Netflix-1M
```
python main.py --dataset Netflix-1M
```
##### Amazon-Music
```
python main.py --dataset Amazon_Digital_Music
```
##### Amazon-Book
```
python main.py --dataset Amazon_Books --layers 2
```

# Settings
We adopt the same experimental configuration as in ReFINe.
The model is built using PyTorch Geometric with a 64-dimensional embedding size and a batch size of 1024.
A 4-layer GCN is used for all datasets, except for Amazon-Book, where a 2-layer GCN is used LightGCN.
The learning rate is set to 1e-3, and training is performed for up to 1000 epochs with early stopping, which is applied if Recall@20 on the validation set does not improve for 50 consecutive epochs.
Following ReFINe, the sampling weight $\gamma$ for confirmed negative feedback is fixed at 1.5.
The regularization coefficient $\lambda_1$ for the $\mathcal{L}_{\text{RW\_SSM}}$ term is set to 1e-7, and $\lambda_2$ for the $\mathcal{L}_{\text{AE}}$ term is set to 1e-5.
The autoencoder for capturing dispreference has one hidden layer and follows the architecture [$|\mathcal{I}|$ → 600 → 64 → 600 → $|\mathcal{I}|$].
In Re-Weighted SSM, we do not treat the number of negative samples as a tunable hyperparameter.
%Instead, it is determined based on a fixed ratio—one-tenth of the total number of items—to minimize dependence on arbitrary hyperparameter tuning.
All experiments are conducted on a single NVIDIA RTX A6000 GPU.

# Compare with our results
The **results4comparison** folder contains the results of our experiment.
Each file includes the loss and performance metrics for every epoch, as well as the hyperparameters, dataset statistics, and training time.
You can compare our results with your own reproduced results.

# Acknowledgments
This work was supported by the Institute of Information \& Communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)], [IITP-2025-RS-2024-00438056, ITRC(Information Technology Research Center) Support Program]. This research was also supported by the Chung-Ang University Research Scholarship Grants in 2023. 
