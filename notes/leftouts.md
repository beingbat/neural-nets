### Parameter Tuning

Parameter tuning methods include (as mentioned in the book as well):

- Add Early Stopping (No improvement in N epochs)
- LR Scheduling (/2 the LR) when no improvement in N epochs happen or after fix number of epochs
- Learning rate as function of epoch numbers

These are not implemented as they are trivial

### Data Augmentation

Improves the model perfomance by virtually expanding data and introducing variations (removing biases in data). opencv-python provides manyß image transformations that could be used. Also, augmentation are not so difficult to implement by hand but their implementation is not directly related to neural networks so skipping them.


### Input Standardizations

Standardization means subtracting mean and dividing by standard deviation. Usually inputs are standardized based on the overall training data mean, std or on the current batch's mean, std (which is considered to be near the whole data's distribution). Then at the inference time, test image is also standardized based on the training time mean, std. This or some other type of standardization i.e. normalizing input between -1 to 1, may help in improving model accuracy. 