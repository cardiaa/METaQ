METaQ (Minimizing Entropy Training and Quantization) is a neural network compression strategy applied during training by adding a regularization term ϕ(w) to the loss function, aiming to minimize the entropy of the network's weights. ϕ(w) results from a non-differentiable optimization problem that is computationally complex to solve. However, to train the network, it is not necessary to explicitly compute ϕ(w); instead, providing its (sub)gradient to standard machine learning tools (such as PyTorch) is sufficient to guide the training towards a low-entropy weight configuration while maintaining good accuracy. This subgradient is computed using the optimal Lagrange multipliers β∗ associated with the set of constraints involving the weights w in the problem that defines ϕ(w).

In this work, we develop a procedure to find β∗ through the application of Lagrangian relaxation techniques and optimization, devising ad hoc methods for specific subproblems when necessary for efficiency reasons. Once the network is trained with low entropy, the compression strategy culminates in quantizing the weights to the reference values of the buckets where the training has directed them. The weight encoding task is handled by well-known compression algorithms that achieve entropy coding.

Experiments were conducted on the LeNet-5 network using the MNIST dataset, though the strategy is applicable to larger networks as well.

The METaQ main directory is structured as follows:

- utils: Contains a set of utility functions used by the various main programs.
- BestModelsBeforeQuantization: Stores trained low-entropy models ready for quantization and compression.
- BestModelAfterQuantization: Stores the final quantized models. These are not compressed, so they can be used for inference.
- notebooks: Contains several Jupyter notebooks showcasing outputs of some instances of the main programs (see below).

Main Programs:
- train.py: Trains the network using the METaQ term.
- load_and_quantize.py: Loads the trained network from train.py, quantizes, and compresses it.
- complexity.py: Runs multiple simulations of the knapsack_specialized algorithm to evaluate its complexity.

A detailed explanation of the entire implemented compression process can be found in the report.pdf file.

  