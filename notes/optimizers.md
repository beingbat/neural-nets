### SGD with `Momentum`
<br/>

$lambda=constant$ `between` $0.0$ to $1.0$ 
<br/>
$V(t) = alpha * ((1-beta) * dW + beta*V(t-1))$
<br/>
$W(t+1) = W(t) - V(t)$
<br/><br/>
There is also another non-intuitive formula of sgd momentum:<br/>
$V(t) = alpha * dW + beta*V(t-1)$
<br/><br/>

### ADAM

Initialize $m = 0, v = 0$ for all weights, biases

$beta1 = constant$

$beta2 = constant$

$alpha = learning\_rate$

$m(t) = beta1 * m(t-1) + (1 – beta1) * dW$

$v(t) = beta2 * v(t-1) + (1 – beta2) * dW^2$

$adjusted = alpha * sqrt(1 – beta2) / (1 – beta1)$

$W(t) = W(t-1) – adjusted * m(t) / (sqrt(v(t)) + eps)$

$t$ denotes number of iterations done i.e. how many times backpropagation is performed.

Decay $beta1$ and $beta2$: (Not sure should it be done after epoch or after each `iteration` of an `epoch`, online sources suggest after each `iteration` but i update them after each `epoch`):

$beta1(t) = beta1(t=0)^t$

$beta2(t) = beta2(t=0)^t$
