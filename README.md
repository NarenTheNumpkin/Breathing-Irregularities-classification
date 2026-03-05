# Usage

## Environment setup

```bash
git clone https://github.com/NarenTheNumpkin/Breathing-Irregularities-classification.git
cd Breathing-Irregularities-classification

conda create --name BIC python=3.12.2 -y
conda activate BIC
pip install -r requirements.txt
```

## Scripts

Generating visualizations `python -m scripts.vis -name "Data/<SUBJECT>"` 

Creating dataset `python -m scripts.create_dataset -in_dir "Data" -out_dir "Dataset"`

Training the model `python -m scripts.train_model`

All visualizations including Plots, Confusion matrices, Metric report can be found in the Visualizations folder. 

# Training Info

Epochs: 20

Batch Size: 32

Optimizer: Adam with 1e-3 lr

Loss function: CrossEntropy

GPU: MPS

# Acknowledgment 

I have taken the help of Gemini to write some of the code.

