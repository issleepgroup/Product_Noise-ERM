# Differentially Private MNIST Training with Opacus

This project implements a simple convolutional neural network (CNN) trained on the MNIST dataset with optional **differential privacy** using [Opacus](https://github.com/pytorch/opacus), a library developed by Meta for training PyTorch models with differential privacy.

The model used is `SampleConvNet`, a lightweight 2-layer CNN. The code allows toggling between standard training and differentially private training using configurable parameters.

---

## Requirements

- Python 3.12
- PyTorch
- Opacus
- torchvision
- tqdm
- pandas

Install dependencies:

```bash
pip install torch torchvision opacus tqdm pandas
```

---

## How to Run

```bash
python mnist.py
```

## Key Arguments

| Argument                        | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `--batch-size`                 | Batch size for training (default: 256)                                     |
| `--test-batch-size`            | Batch size for testing (default: 1024)                                     |
| `--epochs`                     | Number of training epochs (default: 30)                                    |
| `--lr`                         | Learning rate (default: 0.15)                                              |
| `--sigma`                      | Noise multiplier for DP (default: 1.3)                                     |
| `--max-per-sample-grad-norm`   | Gradient clipping norm (default: 1.0)                                      |
| `--delta`                      | Delta parameter for DP (default: 1e-5)                                     |
| `--epsilon`                    | Initial epsilon value (default: 0.3)                                       |
| `--k`                          | Number of training samples (default: 30000)                                |
| `--disable-dp`                 | Use this flag to disable DP and run standard SGD                           |
| `--device`                     | Training device (`cpu` or `cuda`)                                          |

---

## Output

After training, the script outputs a CSV file containing:

- Epoch-wise training accuracy
- Epoch-wise test accuracy
- Epoch-wise epsilon

Example file name:
```
mnist_0.15_1.3_1.0_256_30_0.3_30000.csv
```

---

## Differential Privacy

This implementation uses [Opacus](https://opacus.ai/) to ensure that individual samples' gradients are clipped and noised, offering strong DP guarantees. The privacy accountant is based on **Rényi Differential Privacy (RDP)**.

When DP is enabled, the script prints the cumulative privacy loss ε at each epoch.

---

## Model Architecture

The model is a simple CNN with:

- Two convolutional layers
- Two fully connected layers
- ReLU activation
- Max pooling

Input: 28x28 grayscale MNIST images  
Output: 10-class classification logits

---

## File Structure

```
.
project-root/
├── mnist/                    
│   └── mnist_*.csv           
├── mnist.py                  
└── README.md                 
          
```

---

## License

This project includes code licensed under the **Apache License 2.0**, originally authored by Meta Platforms, Inc.  
You must retain the license information when modifying or redistributing the code.

See [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) for full details.
