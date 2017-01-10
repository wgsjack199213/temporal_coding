### Temporal Coding Using the response Rroperties of Spiking Neuron


##### Implementation Structure
The implementation is organized in 3 classes
- Node: Model of theta neuron
  * theta_der: theta incremental
  * model: simulation of neuron model
- Net: Construct auto-encode network
  * add_layer: construct network architecture layer by layer
  * stimulate: simulation by layer level, with input from previous layer
- SparseCoding/PCA: implemention of application
  * delta_t_delta_w: compute partial derivative
  * forward: forward propagation
  * shuffle_data: divide dataset to training set and validation set
  * learn: network learning   

##### How to Run the code:

pca.py: (Reproduce partial pca)
```python
python pca.py train 
python pca.py retrain [model]
python pca.py test [model] 
```
sparse_encoding.py: (Failure in convergence)
```python 
python sparse_encoding.py train [mu(learning_rate)] [lambda]
python sparse_encoding.py retrain [mu(learning_rate)] [lambda] [model]
python sparse_encoding.py test [model]
```
