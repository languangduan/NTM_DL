# NTM_DL: Neural Topic Model with Deep Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)

## 📝 Description

This repository contains an implementation of the paper ["A Novel Neural Topic Model and Its Supervised Extension"](https://ojs.aaai.org/index.php/AAAI/article/view/9361) (AAAI 2015) as part of the AI66103 course project.

### Paper Citation
```bibtex
@inproceedings{cao2015novel,
  title={A Novel Neural Topic Model and Its Supervised Extension},
  author={Cao, Ziqiang and Li, Sujian and Liu, Yang and Li, Wenjie and Ji, Heng},
  booktitle={Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence},
  pages={2210--2216},
  year={2015}
}
```

## ✨ Features

- PyTorch implementation of Neural Topic Model (NTM)
- Support for text classification and topic modeling tasks
- Includes preprocessed datasets (e.g., Wiki10)

## 🚀 Quick Start

### Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

### Installation & Running

1. Clone the repository:
```bash
git clone https://github.com/languangduan/NTM_DL.git
cd NTM_DL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure parameters:
- Open `utils/utils.py`
- Modify the command line arguments according to your needs
- Key parameters include:
  - Dataset path
  - Model parameters
  - Training parameters

4. Run the model:
```bash
python run.py
```

## 📊 Datasets

The project supports the following preprocessed datasets:
- Wiki10 dataset
- MovieReviewDtaset
- 20NewsDataset

To use datasets, please download manually and extract them in `dataset/` path.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

- **Course**: AI6103
- **Assignment**: Paper Implementation
- **Posted by:** CCDS Li Boyang
- **Original Paper**: "A Novel Neural Topic Model and Its Supervised Extension" (AAAI 2015)
- **Authors**: Ziqiang Cao, Sujian Li, Yang Liu, Wenjie Li, Heng Ji
