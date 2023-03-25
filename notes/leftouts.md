### Parameter Tuning

Parameter tuning methods include (as mentioned in the book as well):

- Add Early Stopping (No improvement in N epochs)
- LR Scheduling (/2 the LR) when no improvement in N epochs happen or after fix number of epochs
- Learning rate as function of epoch numbers

These are not implemented as they are trivial

### Data Augmentation

Improves the model perfomance by virtually expanding data and introducing variations (removing biases in data). opencv-python provides manyß image transformations that could be used. Also, augmentation are not so difficult to implement by hand but their implementation is not directly related to neural networks so skipping them.