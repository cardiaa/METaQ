METaQ (Minimizing Entropy Training and Quantization) is a neural network compression strategy applied during training by adding a regularization term $\phi(w)$ to the loss function, aiming to minimize the entropy of the network's weights. $\phi(w)$ results from a non-differentiable optimization problem that is computationally complex to solve. However, to train the network, it is not necessary to explicitly compute $\phi(w)$; instead, providing its (sub)gradient to standard machine learning tools (such as PyTorch) is sufficient to guide the training towards a low-entropy weight configuration while maintaining good accuracy. This subgradient is computed using the optimal Lagrange multipliers $\beta^*$ associated with the set of constraints involving the weights w in the problem that defines $\phi(w)$.

In this work, we develop a procedure to find $\beta^*$ through the application of Lagrangian relaxation techniques and optimization, devising ad hoc methods for specific subproblems when necessary for efficiency reasons. Once the network is trained with low entropy, the compression strategy culminates in quantizing the weights to the reference values of the buckets where the training has directed them. The weight encoding task is handled by well-known compression algorithms that achieve entropy coding.

Experiments were conducted on the LeNet-5 network using the MNIST dataset, though the strategy is applicable to larger networks as well.

The METaQ main directory is structured as follows:

- BestModelsBeforeQuantization: Stores trained low-entropy models ready for quantization and compression.
- BestModelAfterQuantization: Stores the final quantized models. These are not compressed, so they can be used for inference.
- notebooks: Contains several Jupyter notebooks showcasing outputs of some instances of the main programs (see below). In the notebook Training.ipynb we train the network with the METaQ strategy; if performances are better than those already reached we save the model in the BestModelsBeforeQuantization folder. In the notebook Load&Compression.ipynb we load the model from those created during the training phase and we compress it after having quantize it. In the notebook CuttingPlaneAlgo.ipynb we test all the algorithms to solve the simil-knapsack problem by evaluating their correctness and their performance. In the notebook ComplexityAnalysis.ipynb we evaluate the complexity of the specialized algorithm for the simil-knapsack problem form a probabilistic point of view. Finally, in the notebook SubgradientBundleMethods.ipynb we evaluate and test the algorithms for solve the dual problem of maximization $\phi_w(\xi)$.

Besides the notebook one can run the code with the folder organization as follow.

- utils: Contains a set of utility functions used by the various main programs.

Main Programs:
- train.py: Trains the network using the METaQ term.
- load_and_quantize.py: Loads the trained network with train.py, quantizes, and compresses it.

A detailed explanation of the entire implemented compression process can be found in the Report.pdf file.

  