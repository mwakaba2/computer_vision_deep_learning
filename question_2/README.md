## Question 2: Simple Neural Networks

Assume we have a fully-connected neural network that does image recognition with one hidden layer where the input is an image with channels R,G,B, and dimensions 224x224.
```javascript
    The input is X, W1 is weight 1, B1 is bias 1, W2 is weight 2, B2 is bias 2, and y is the output.
    The output is y = choose_max_index(softmax(W2 x ReLU(W1 x X + B1) + B2))
    X: N x D, N is the number of samples, D is 3x224x224
    W1: D x H, D is 3x224x224, H the hidden layer size
    B1: 1 x H, a vector of size H
    ReLU: np.maximum(value, 0)
    W2: H x C, H is the hidden layer size, C the number of output labels
    B2: 1 x C, vector of size C
    softmax: e^z_j/Sum(e^x_i) for i = 1, ... C; for j in 1 to C
    choose_max_index: for all values of softmax function, choose the index with the max output
```
1. What would be an appropriate loss function for this neural network? How would you ensure that the network doesn't overfit?
2. Explain how you would train the network to find the appropriate weights and biases. Be sure to include details on what the gradients are for W1, W2, B1, and B2 if you are using back-propagation. Write the pseudo code for your training process.


### Answers

#### Question 1
**a. What would be an appropriate loss function for this neural network?**

This neural network will learn the parameters quickly with the cross-entropy loss function. The neural network will converge faster with cross-entropy function than the mean squared error function. If we use MSE, the gradients calculated during back propagation will quickly get smaller, and will start to converge slowly. In contrast, the weight changes will not get smaller with cross-entropy.

**b. How would you ensure that the network doesn't overfit?**

A fully connected neural network can overfit because of the number of parameters it has.
There are several ways to avoid overfitting:
* We can add more images or use data augmentation to increase the training dataset. This will help the model generalize better.
* We can avoid overfitting with regularization parameters. The L2 regularization can decrease the weights in the loss function.
* Another technique would be to use dropout. While training, the network will "forget" and learn from random neurons instead of learning from all neurons.
* We can also use early-stopping to stop the network training when the validation error surpasses the training error. This can help reduce training time.

#### Question 2
I'll use mini-batch gradient descent to find the appropriate weights and biases for this network.
##### Mini-batch gradient descent pseudocode
<pre><code>mini batch size = N
For each training batch t
	1. Forward propagation on X<sup>{t}</sup>
	2. Compute loss
	3. Backpropagate to compute gradients with respect to loss function
	4. Update weights and biases
</code></pre>

#### 1. Forward propagation on X<sup>{t}</sup>
<pre><code>Z<sup>[1]</sup> = W<sup>[1]</sup>X<sup>{t}</sup>+b<sup>[1]</sup>
A<sup>[1]</sup> = g<sup>[1]</sup>(Z<sup>[1]</sup>) where g = ReLU activation
Z<sup>[2]</sup> = W<sup>[2]</sup>A<sup>[1]</sup>+b<sup>[2]</sup>
A<sup>[2]</sup> = g<sup>[2]</sup>(Z<sup>[2]</sup>) where g = Softmax activation
</code></pre>

#### 2. Compute loss
<pre><a href="https://www.codecogs.com/eqnedit.php?latex=J^{\{t\}}(W^{[1]}&space;,&space;b^{[1]},&space;W^{[2]},&space;b^{[2]})&space;=\frac{1}{N}&space;\sum_{i=1}^{l}L(\widehat{y}^{(i)},&space;y^{(i)})&space;\textup{&space;from&space;}&space;x^{\{t\}},&space;y^{\{t\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J^{\{t\}}(W^{[1]}&space;,&space;b^{[1]},&space;W^{[2]},&space;b^{[2]})&space;=\frac{1}{N}&space;\sum_{i=1}^{l}L(\widehat{y}^{(i)},&space;y^{(i)})&space;\textup{&space;from&space;}&space;x^{\{t\}},&space;y^{\{t\}}" title="J^{\{t\}}(W^{[1]} , b^{[1]}, W^{[2]}, b^{[2]}) =\frac{1}{N} \sum_{i=1}^{l}L(\widehat{y}^{(i)}, y^{(i)}) \textup{ from } x^{\{t\}}, y^{\{t\}}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{For&space;any&space;}c&space;\in&space;\{1...C\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{For&space;any&space;}c&space;\in&space;\{1...C\}" title="\textup{For any }c \in \{1...C\}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=L(\widehat{y}^{(i)},&space;y^{(i)})&space;=&space;\begin{cases}&space;-log(\widehat{y})&space;&&space;\text{&space;if&space;}&space;\widehat{y}=&space;c\\&space;-log(1-\widehat{y})&space;&&space;\text{&space;if&space;}&space;\widehat{y}\neq&space;c\\&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\widehat{y}^{(i)},&space;y^{(i)})&space;=&space;\begin{cases}&space;-log(\widehat{y})&space;&&space;\text{&space;if&space;}&space;\widehat{y}=&space;c\\&space;-log(1-\widehat{y})&space;&&space;\text{&space;if&space;}&space;\widehat{y}\neq&space;c\\&space;\end{cases}" title="L(\widehat{y}^{(i)}, y^{(i)}) = \begin{cases} -log(\widehat{y}) & \text{ if } \widehat{y}= c\\ -log(1-\widehat{y}) & \text{ if } \widehat{y}\neq c\\ \end{cases}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{y}^{(i)}&space;=&space;A^{[2]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{y}^{(i)}&space;=&space;A^{[2]}" title="\widehat{y}^{(i)} = A^{[2]}" /></a>
</code></pre>

#### 3. Backpropagation
<pre><code>
<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{For&space;layer&space;}l&space;\in&space;[1,&space;2]&space;\textup{&space;}&space;dW^{[l]}=\frac{\partial&space;J}{\partial&space;W^{[l]}},&space;db^{[l]}=\frac{\partial&space;J}{\partial&space;b^{[l]}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{For&space;layer&space;}l&space;\in&space;[1,&space;2]&space;\textup{&space;}&space;dW^{[l]}=\frac{\partial&space;J}{\partial&space;W^{[l]}},&space;db^{[l]}=\frac{\partial&space;J}{\partial&space;b^{[l]}}" title="\textup{For layer }l \in [1, 2] \textup{ } dW^{[l]}=\frac{\partial J}{\partial W^{[l]}}, db^{[l]}=\frac{\partial J}{\partial b^{[l]}}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;dZ^{[2]}&space;=&space;A^{[2]}&space;-&space;Y^{\{t\}}&space;\newline&space;dW^{[2]}&space;=&space;\frac{1}{N}&space;dZ^{[2]}A^{[1]}T&space;\newline&space;db^{[2]}&space;=&space;\sum_{i=1}^{N}&space;dZ^{[2]}&space;\newline\newline&space;dZ^{[1]}&space;=&space;W^{[2]}TdZ^{[2]}&space;*&space;{g^{[1]}}'(Z^{[1]})\newline&space;dW^{[1]}&space;=&space;\frac{1}{N}&space;dZ^{[1]}X^{\{t\}}T&space;\newline&space;db^{[1]}&space;=&space;\sum_{i=1}^{N}&space;dZ^{[1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;dZ^{[2]}&space;=&space;A^{[2]}&space;-&space;Y^{\{t\}}&space;\newline&space;dW^{[2]}&space;=&space;\frac{1}{N}&space;dZ^{[2]}A^{[1]}T&space;\newline&space;db^{[2]}&space;=&space;\sum_{i=1}^{N}&space;dZ^{[2]}&space;\newline\newline&space;dZ^{[1]}&space;=&space;W^{[2]}TdZ^{[2]}&space;*&space;{g^{[1]}}'(Z^{[1]})\newline&space;dW^{[1]}&space;=&space;\frac{1}{N}&space;dZ^{[1]}X^{\{t\}}T&space;\newline&space;db^{[1]}&space;=&space;\sum_{i=1}^{N}&space;dZ^{[1]}" title="\newline dZ^{[2]} = A^{[2]} - Y^{\{t\}} \newline dW^{[2]} = \frac{1}{N} dZ^{[2]}A^{[1]}T \newline db^{[2]} = \sum_{i=1}^{N} dZ^{[2]} \newline\newline dZ^{[1]} = W^{[2]}TdZ^{[2]} * {g^{[1]}}'(Z^{[1]})\newline dW^{[1]} = \frac{1}{N} dZ^{[1]}X^{\{t\}}T \newline db^{[1]} = \sum_{i=1}^{N} dZ^{[1]}" /></a>

</code></pre>

#### 4. Update Weights
<pre><code>
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\textup{&space;is&space;the&space;learning&space;rate}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\textup{&space;is&space;the&space;learning&space;rate}" title="\alpha \textup{ is the learning rate}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{For&space;layer&space;}l&space;\in&space;[1,&space;2]&space;\textup{&space;}&space;W^{[l]}=W^{[l]}&space;-&space;\alpha&space;dW^{[l]},&space;b^{[l]}=b^{[l]}&space;-&space;\alpha&space;db^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{For&space;layer&space;}l&space;\in&space;[1,&space;2]&space;\textup{&space;}&space;W^{[l]}=W^{[l]}&space;-&space;\alpha&space;dW^{[l]},&space;b^{[l]}=b^{[l]}&space;-&space;\alpha&space;db^{[l]}" title="\textup{For layer }l \in [1, 2] \textup{ } W^{[l]}=W^{[l]} - \alpha dW^{[l]}, b^{[l]}=b^{[l]} - \alpha db^{[l]}" /></a>
</code></pre>
