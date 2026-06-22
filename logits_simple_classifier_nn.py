# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "torchviz==0.0.3",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim

    return mo, nn, optim, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Define the network
    """)
    return


@app.cell
def _(nn):
    # A simple linear layer: 1 input feature -> 3 output nodes (one for each class)
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super(SimpleClassifier, self).__init__()
            self.linear = nn.Linear(1, 3)
        
        def forward(self, x):
            # Returns raw, unnormalized scores (Logits)
            return self.linear(x)

    return (SimpleClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##2. Setup examples
    """)
    return


@app.cell
def _(torch):
    # X: Numbers from 0 to 15
    # Y: Classes (0: x < 5, 1: 5 <= x <= 10, 2: x > 10)
    X_train = torch.tensor([[float(i)] for i in range(16)], dtype=torch.float32)
    Y_train = torch.tensor([0 if x < 5 else 1 if x <= 10 else 2 for x in X_train], dtype=torch.long)
    return X_train, Y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Initialize Model, Loss, and Optimizer
    """)
    return


@app.cell
def _(SimpleClassifier, nn, optim):
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss() # Note: CrossEntropyLoss expects raw logits
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    return criterion, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train the Model
    """)
    return


@app.cell
def _(X_train, Y_train, criterion, model, optimizer):
    print("Training model...")
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
    print("Training complete!\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Demo the Logits vs. Softmax
    """)
    return


@app.cell
def _(model, torch):
    print("--- DEMO: Logits vs. Softmax ---")
    model.eval()

    # Let's test three distinct numbers
    test_numbers = [2, 7.5, 13]

    with torch.no_grad():
        for num in test_numbers:
            input_tensor = torch.tensor([[num]], dtype=torch.float32)
        
            # Step A: Get raw Logits
            logits = model(input_tensor)
        
            # Step B: Apply Softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)
        
            # Step C: Get the predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
            print(f"Input Number: {num}")
            print(f"  Raw Logits:     {logits.numpy()[0]}")
            print(f"  Softmax Probabilities:  {[float(x) for x in probabilities.numpy()[0]]}")
            print(f"  Predicted Class:            {predicted_class}")
            print("-" * 40)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Verify
    """)
    return


@app.cell
def _(model):
    import numpy as np

    # Put model in eval mode
    model.eval()

    # Extract parameters as NumPy arrays
    # PyTorch stores weights as (out_features, in_features), i.e., Shape (3, 1)
    W = model.linear.weight.detach().numpy()
    b = model.linear.bias.detach().numpy()

    print("--- Extracted Model Parameters ---")
    print(f"Weight Matrix W (Shape {W.shape}):\n{W}\n")
    print(f"Bias Vector b (Shape {b.shape}):\n{b}\n")
    return W, b, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $\text{Logits} = X \cdot W^T + b$
    """)
    return


@app.cell
def _(W, b, model, np, torch):
    test_val = 7.5
    X_sample = np.array([[test_val]])

    # Get PyTorch's Official Logits
    with torch.no_grad():
        torch_input = torch.tensor(X_sample, dtype=torch.float32)
        torch_logits = model(torch_input).numpy()[0]

    # We transpose W to Shape (1, 3) to match our equation
    manual_logits = np.dot(X_sample, W.T) + b
    manual_logits = manual_logits[0] # Flatten to a 1D array for easy comparison

    # 4. Print and Compare
    print("--- Logits Recomputation Verification ---")
    print(f"Input Number: {test_val}")
    print(f"PyTorch Official Logits: {torch_logits}")
    print(f"Manual Recomputed Logits: {manual_logits}")

    # Check if they match perfectly
    if np.allclose(torch_logits, manual_logits, atol=1e-5):
        print("\n Success! The manual math perfectly matches PyTorch's output.")
    else:
        print("\n Discrepancy found. Check matrix transpositions.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $W = \begin{bmatrix} -2.1313477 \\ -0.17493701 \\  0.92801386 \end{bmatrix}$, $b = \begin{bmatrix} 11.314914 \\ 2.6907895 \\ -8.756229 \end{bmatrix}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\begin{bmatrix} \text{Logit}_0 \\ \text{Logit}_1 \\ \text{Logit}_2 \end{bmatrix} = \begin{bmatrix} -2.1313477 \\ -0.17493701 \\  0.92801386 \end{bmatrix} [x] + \begin{bmatrix} 11.314914 \\ 2.6907895 \\ -8.756229 \end{bmatrix}$
    """)
    return


@app.cell
def _():
    -2.1313477 * 7.5 + 11.314914
    return


if __name__ == "__main__":
    app.run()
