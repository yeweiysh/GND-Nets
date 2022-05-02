# GNDC

## Overview
Here we provide the implementation of GND-Nets (Graph Neural Diffusion Networks) in TensorFlow. The repository is organised as follows:
- `data/` contains datasets Cora, Cora-ML, Citeseer, Pubmed, Amazon Computers, and Amazon Photo;
- `new_data/` contains datasets Chameleon and Squirrel;
- `models/` contains the implementation of the GND-Nets (`gndnets_slp.py`, `gndnets_mlp.py`, and `gndnets_ds.py`);
- `utils/` contains:
    * an implementation of three variants of graph neural diffusions (`layers.py`);
    * preprocessing subroutines (`process.py`);

Finally, `bash run_train` execute the experiments.


## Dependencies

The script has been tested running under Python 3.7.9, with TensorFlow version as:
- `tensorflow==2.6.0`

In addition, CUDA 11.4 and cuDNN 11.1 have been used.


## License
MIT
