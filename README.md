## Description

I used pytorch-lighting framework for training.

## Usage
### Miniconda

If not installed miniconda in desktop, use command for install.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

```bash
conda create -n level1 python=3.10 -y
conda activate level1
pip install -r requirements.txt
```

### Train

before train, login 'wandb' first.
```bash
wandb login
```

```python
python main.py -o pl_train
```

### Inference
```python
python main.py -o pl_inference
```

### Ensemble
```python
python main.py -o ensemble
```
