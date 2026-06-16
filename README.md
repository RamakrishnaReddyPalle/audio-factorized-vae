## **Installation**

Create environment:

```bash
conda create -n afvae python=3.11
conda activate afvae
```

Install PyTorch for your hardware:

### **CUDA**

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
```

### **CPU**

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```