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


# Compare with our results
The **results4comparison** folder contains the results of our experiment.
Each file includes the loss and performance metrics for every epoch, as well as the hyperparameters, dataset statistics, and training time.
You can compare our results with your own reproduced results.

# Acknowledgments

