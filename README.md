# XOR Neural Network from Scratch (NumPy)

This project implements a simple feed-forward neural network **from scratch using only NumPy** to solve the classic **XOR problem**.  

The network:
- Uses **1 hidden layer** with sigmoid activation.
- Trains using **binary cross-entropy loss**.
- Implements **forward propagation** and **backpropagation** manually.
- Achieves **near-zero training loss** and **100% accuracy** on clean test data.

--------------------------------------------------------------------------------------------

## Features
- **Custom sigmoid & derivative**
- **Weight initialization (with Xavier normalization)**
- **Forward propagation**
- **Backpropagation with gradient descent**
- **Cross-entropy loss with clipping for numerical stability**
- **Training & evaluation on noisy and clean XOR datasets**

--------------------------------------------------------------------------------------------

## Results

- **Training Loss**: Converges to nearly **0**
- **Test Accuracy**: **1.0** (perfect classification on clean XOR dataset)

Training log:
```
At iteration 0, loss = 0.6926197475619172
At iteration 500, loss = 0.6918384136262908
At iteration 1000, loss = 0.6916363837837659
At iteration 1500, loss = 0.6914298172178107
At iteration 2000, loss = 0.6889351654213592
At iteration 2500, loss = 0.44934424702620396
At iteration 3000, loss = 0.09053926446905555
At iteration 3500, loss = 0.03288802227877012
At iteration 4000, loss = 0.017881369554036332
```

Final metrics:
- **Training Loss**: ~0.018
- **Test Accuracy**: 1.0

--------------------------------------------------------------------------------------------

## Visualizations

### Training Loss Curve
Training loss decreases smoothly to near zero:

![Loss Curve](loss_curve.png)

--------------------------------------------------------------------------------------------

### Decision Boundary
The trained network learns the correct XOR classification regions:

![Decision Boundary](decision_boundary.png)

--------------------------------------------------------------------------------------------

## Model Architecture
- **Input Layer**: 2 neurons (XOR inputs)
- **Hidden Layer**: 14 neurons, sigmoid activation
- **Output Layer**: 1 neuron, sigmoid activation

--------------------------------------------------------------------------------------------

## Dataset
Two datasets are generated:
1. **Training Set (with noise)** → Slightly perturbed XOR inputs  
2. **Test Set (clean)** → Perfect XOR mapping (no noise)

--------------------------------------------------------------------------------------------

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/MohitAryal/Neural-Network-from-scratch-using-numpy.git
   cd Neural-Network-from-scratch-using-numpy
   ```

2. Run training:
   ```bash
   python main.py
   ```

3. Expected Output:
   ```
   At iteration 0, loss = 0.6926
   ...
   Final Test Accuracy: 1.0
   ```

--------------------------------------------------------------------------------------------

## 📦 Requirements
- Python 3.8+
- NumPy
- Matplotlib (for plotting)

Install dependencies:
```bash
pip install numpy matplotlib
```

--------------------------------------------------------------------------------------------

## Plotting the Loss Curve

Add after training:
```python
import matplotlib.pyplot as plt

plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()
```

--------------------------------------------------------------------------------------------

## Visualizing Decision Boundary

After training, add:

```python
import matplotlib.pyplot as plt

# Generate grid
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Forward pass
_, Z = forward_propagation(learned_w1, learned_b1, learned_w2, learned_b2, grid)
Z = Z.reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, Z > 0.5, alpha=0.6, cmap=plt.cm.coolwarm)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, cmap=plt.cm.coolwarm, edgecolors="k")
plt.title("Decision Boundary for XOR")
plt.savefig("decision_boundary.png")
plt.show()
```

--------------------------------------------------------------------------------------------

## Key Takeaways
- The XOR problem is **not linearly separable**, requiring a hidden layer to solve.
- Even a simple 2-layer neural network trained with backpropagation can learn XOR perfectly.
- This project demonstrates **how neural networks work under the hood** without relying on deep learning frameworks.

--------------------------------------------------------------------------------------------