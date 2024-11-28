# constrained_learning
Handy package for training output constrained neural networks. As of now, this package only contains 
constrained extreme learning machines (CELM), constrained multi layer neural networks will be added in the future.

# Structure
The source code is contained in the [constrained_learning](constrained_learning) folder.
The implementation of the (C)ELM can be found in [constrained_learning/learner.py](constrained_learning/learner.py).

To install the package, run the following commands inside this repository:
```
conda create -n celm python=3.10
conda activate celm
pip install -e .
```

To run the synthetic examples: 
``` 
python constrained_learning\examples\flat_gaussian.py
python constrained_learning\examples\monotone_quadratic.py
python constrained_learning\examples\ode.py
```

# License
MIT license - See [LICENSE](LICENSE).   
