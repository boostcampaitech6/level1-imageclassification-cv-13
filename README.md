## Description

Pytorch baseline code

## Usage
### Miniconda

If not installed miniconda in desktop, use command for install
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

```bash
pip install virtualenv
cd ${PROJECT}
python -m venv /v2/venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
wandb login
```
```bash
python train.py
```

### Pretrained

```bash
python train.py --pretrained
```

