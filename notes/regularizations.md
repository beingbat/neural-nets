## Regularizations

#### L1/L2 REGULARIZATIONS

L1 and L2 Regularizations are added.

For backpropagation, loss function is broken down into two parts. Part one's gradient is found using chain rule and part two's gradient is found directly as it is direct derivative of weight(s).

Regularization is not performed on bias by convention

#### DROPOUT

To implement dropout, it must be remembered that which weights did not participate in forward pass so they don't get updated in the backward pass. This introduces complexity of maintaining an extra list / mask for weights.