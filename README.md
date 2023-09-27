# Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks

## Dependencies and Installation
1. **Before Start:** make sure your cuda version is 11.3+.
2. **Clone the GitHub Repository:**
   ```
   git clone git@github.com:ccffccffcc/stochastic_collapse.git
   ```
3. **Create Conda Environment:** It is reommended to Create a New Virtual Environment. You can replace `stochastic_collapse` with your preferred environment name.
   ```
   conda create -y --name stochastic_collapse python=3.10
   conda activate stochastic_collapse
   ```
4. **Install Dependencies:** Please make sure you install the correct version of Pytorch and MosaicML.
   ```
   conda install -c conda-forge cudatoolkit
   pip3 install torch==1.11.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   pip install --upgrade mosaicml==0.7.1
   pip install black GPUtil isort ipython jupyter matplotlib pandas python-dotenv seaborn scipy wandb ffcv numba opencv-python cupy-cuda113
   pip install -e .
   ```
5. **Test installation**: Please set environment variable `DATADIR` to the desired directory for dataset. You would either start to download or verify the CIFAR dataset.
   ```
   python run/make_cifar.py
   ```

## Run Experiment
Assign environment variable `DATA_DIR` and `EXP_DIR` to your data directory and logging saving directory.
To run experiment, for example, you can use the following codes:
   ```
   python run/exp.py train -f configs/cifar10_sgd/train_sce_gelu_resnet.yaml
   ```
To finetune a trained model, for example, you can use the following codes:
   ```
   python run/exp.py finetune -f configs/cifar10_sgd/train_sce_gelu_lbn_final_finetune.yaml
   ```

## Citation

   If you find our results or codes useful for your research, please consider citing our paper:

   ```bibtex
   @misc{chen2023stochastic,
        title={Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks}, 
        author={Feng Chen and Daniel Kunin and Atsushi Yamamura and Surya Ganguli},
        year={2023},
        eprint={2306.04251},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
        }
   ```
