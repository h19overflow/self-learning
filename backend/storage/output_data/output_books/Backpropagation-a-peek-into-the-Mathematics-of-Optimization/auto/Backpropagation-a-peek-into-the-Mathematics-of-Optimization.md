# Backpropagation. A Peek into the Mathematics of Optimization

![## Image Analysis: 478a8caeccfaf7d8aa980909eefff60b532264d45ab9fa285a9b816acb562bd5.jpg

**Conceptual Understanding:**
The image conceptually represents the brand identity of an organization named "365 DataScience". Its main purpose is to brand the associated document or content, indicating its origin or authorship. The key ideas communicated are the organization's name and its primary focus on data science, with implied notions of comprehensiveness and data connectivity through its stylistic elements.

**Content Interpretation:**
This image primarily serves as a brand identifier, specifically the logo for "365 DataScience". It represents a company or educational platform focused on data science. The integration of the "365" with a degree symbol and the network-like graphic suggests themes of comprehensive coverage, interconnected data, or a complete learning journey within the field of data science.

**Key Insights:**
The main takeaway is the identification of the brand "365 DataScience". The logo conveys that this entity is involved in the field of data science, potentially offering educational content or services. The '365' often implies a comprehensive, all-encompassing, or continuous aspect, while 'DataScience' directly states the domain of expertise. The network icon visually reinforces concepts of data connections, algorithms, or complex systems inherent in data science. This strongly suggests that the accompanying document content, concerning 'Backpropagation' and 'Optimization', is provided by or associated with '365 DataScience'.

**Document Context:**
Given the document context "Backpropagation. A Peek into the Mathematics of Optimization," this image functions as a branding element for the entity responsible for the content, likely an educational provider or a platform offering courses related to data science, such as '365 DataScience'. It signifies the source or publisher of the academic/technical material being presented.

**Summary:**
The image displays a corporate logo on a teal background. The logo features the text "365° DataScience" in white, bold sans-serif font. Between "365°" and "DataScience", there is a stylized graphic composed of three white circular nodes connected by white lines, forming an inverted 'V' shape. This graphic visually separates the numerical part "365°" from the textual "DataScience", while also creating a visual link. The overall presentation is clean and professional, characteristic of a brand identity.](images/478a8caeccfaf7d8aa980909eefff60b532264d45ab9fa285a9b816acb562bd5.jpg)

# 1 Motivation

In order to get a truly deep understanding of deep neural networks, one must look at the mathematics of it. As backpropagation is at the core of the optimization process, we wanted to introduce you to it. This is definitely not a necessary part of the course, as in TensorFlow, sk-learn, or any other machine learning package (as opposed to simply NumPy), will have backpropagation methods incorporated.

# 2 The specific net and notation we will examine

Here’s our simple network:

![## Image Analysis: 1a6d08ea3053e54ff5e71bcf79c6bea7bb07a52776b725df216f51a48f4c23b7.jpg

**Conceptual Understanding:**
This image represents a simple, three-layer feedforward artificial neural network. Conceptually, it illustrates how input data is processed through intermediate computational units (hidden neurons) via weighted connections to produce an output. The main purpose of the diagram is to visually define the architecture and notation of a specific neural network model. It communicates the key ideas of layered network structure, the role of synaptic weights in connecting neurons, and the unidirectional flow of information from inputs to outputs, which forms the 'forward pass' of network operation. The presence of target values ('t1', 't2') alongside the network outputs ('y1', 'y2') suggests its use in a supervised learning context where the network learns to map inputs to desired outputs by adjusting its internal weights.

**Content Interpretation:**
The image displays a feedforward artificial neural network architecture with two input neurons (x1, x2), one hidden layer consisting of three neurons (h1, h2, h3), and two output neurons (y1, y2). The network's outputs (y1, y2) are shown leading to target values (t1, t2).

Processes shown:
*   **Input Reception:** The network receives two input signals, 'x1' and 'x2'.
*   **Weighted Summation at Hidden Layer:** Each input 'x_i' is multiplied by a weight 'w_ij' before being passed to a hidden neuron 'h_j'. Specifically:
    *   'x1' connects to 'h1' via weight 'w11', to 'h2' via 'w12', and to 'h3' via 'w13'.
    *   'x2' connects to 'h1' via weight 'w21', to 'h2' via 'w22', and to 'h3' via 'w23'.
*   **Hidden Layer Activation (implicit):** The hidden neurons 'h1', 'h2', 'h3' process the weighted sum of their inputs.
*   **Weighted Summation at Output Layer:** The output of each hidden neuron 'h_j' is multiplied by a weight 'u_jk' before being passed to an output neuron 'y_k'. Specifically:
    *   'h1' connects to 'y1' via weight 'u11' and to 'y2' via 'u12'.
    *   'h2' connects to 'y1' via weight 'u21' and to 'y2' via 'u22'.
    *   'h3' connects to 'y1' via weight 'u31' and to 'y2' via 'u32'.
*   **Output Layer Activation (implicit):** The output neurons 'y1', 'y2' produce the network's predictions.
*   **Target Comparison:** The network's outputs 'y1' and 'y2' are compared against external target values 't1' and 't2'.

Relationships:
*   **Hierarchical Layered Structure:** Information flows sequentially from the input layer (x nodes) to the hidden layer (h nodes) and then to the output layer (y nodes).
*   **Fully Connected:** Every neuron in a preceding layer is connected to every neuron in the subsequent layer.
*   **Weighted Connections:** The strength of influence between neurons is determined by specific weights ('w' for input-to-hidden, 'u' for hidden-to-output).

Significance of information:
*   `x1`, `x2`: Represent input features.
*   `w_ij`: Weights determining influence from input layer to hidden layer.
*   `h1`, `h2`, `h3`: Activations of hidden neurons, representing learned features.
*   `u_ij`: Weights determining influence from hidden layer to output layer.
*   `y1`, `y2`: Network's predicted outputs.
*   `t1`, `t2`: Desired target values for comparison with network outputs.

**Key Insights:**
The main takeaways from this image are:
*   **Standard Neural Network Architecture:** It depicts a common feedforward neural network structure with distinct input, hidden, and output layers, demonstrating how neurons are organized. (Evidence: Nodes 'x1', 'x2', 'h1', 'h2', 'h3', 'y1', 'y2' clearly define these layers).
*   **Role of Weighted Connections:** Learning in such networks primarily involves adjusting the synaptic weights (parameters) associated with each connection. The detailed labeling of 'w' and 'u' weights highlights their specific importance in determining information flow and network behavior. (Evidence: Labels 'w11' through 'w23' and 'u11' through 'u32' on all connecting arrows).
*   **Unidirectional Information Flow (Feedforward):** The consistent left-to-right direction of all arrows illustrates the feedforward nature, where information progresses from inputs through the network to outputs without loops or cycles in the primary processing path. (Evidence: All arrows point from x nodes to h nodes, from h nodes to y nodes, and from y nodes to t values).
*   **Supervised Learning Context:** The presence of target values 't1' and 't2' implies that this network is designed for a supervised learning task, where the network's outputs are compared to known correct labels or values to calculate error and guide training. (Evidence: Arrows from 'y1' to 't1' and 'y2' to 't2').
*   **Foundation for Backpropagation:** This diagram provides the exact structure and notation needed to explain and apply the backpropagation algorithm, which will use the error between 'y' and 't' to update the 'w' and 'u' weights iteratively. (Evidence: The complete definition of input, hidden, and output nodes, along with all connecting weights and target values, forms the basis for backpropagation discussion).

**Document Context:**
The image serves as a foundational diagram for the section "2 The specific net and notation we will examine" and is explicitly titled "Figure 1: Backpropagation" in the document context. It visually defines the neural network architecture upon which the backpropagation algorithm would be applied. Backpropagation requires a clear understanding of the network's forward pass, including its inputs, hidden layers, outputs, and all associated weights. This diagram precisely illustrates these components and their notations ('x', 'h', 'y' for neurons; 'w', 'u' for weights; 't' for targets), making it critical for explaining how the error would be calculated and then propagated backward through the network to adjust these weights.

**Summary:**
This diagram illustrates a simple, fully-connected feedforward neural network, a foundational model in machine learning. It's composed of three distinct layers: an input layer, a hidden layer, and an output layer, with information flowing unidirectionally from left to right.

1.  **Input Layer:** The network begins with two input neurons, labeled `x1` and `x2`. These nodes represent the initial data or features fed into the system.

2.  **Connections to the Hidden Layer (Weights 'w'):** Each input neuron is connected to every neuron in the hidden layer. These connections are associated with numerical values called 'weights', which determine the strength and influence of each input on the subsequent layer.
    *   From `x1`, there are connections leading to:
        *   `h1` with weight `w11`
        *   `h2` with weight `w12`
        *   `h3` with weight `w13`
    *   From `x2`, there are connections leading to:
        *   `h1` with weight `w21`
        *   `h2` with weight `w22`
        *   `h3` with weight `w23`
    The first digit in the weight `w_ij` often refers to the input node (`i`), and the second digit to the hidden node (`j`) it connects to.

3.  **Hidden Layer:** This layer consists of three neurons, labeled `h1`, `h2`, and `h3`. These hidden nodes receive the weighted sum of inputs from the `x` nodes and typically apply an activation function to transform this sum. The hidden layer is where the network learns to extract features or patterns from the input data.

4.  **Connections to the Output Layer (Weights 'u'):** Similar to the input-to-hidden layer, each hidden neuron is fully connected to every neuron in the output layer. These connections also have associated 'weights', denoted by 'u'.
    *   From `h1`, there are connections leading to:
        *   `y1` with weight `u11`
        *   `y2` with weight `u12`
    *   From `h2`, there are connections leading to:
        *   `y1` with weight `u21`
        *   `y2` with weight `u22`
    *   From `h3`, there are connections leading to:
        *   `y1` with weight `u31`
        *   `y2` with weight `u32`
    Here, the `u_jk` notation typically means a weight from hidden node `j` to output node `k`.

5.  **Output Layer:** The final layer of the network contains two output neurons, labeled `y1` and `y2`. These nodes produce the network's predictions or final results based on the processed information from the hidden layer.

6.  **Target Values:** The arrows extending from `y1` to `t1` and `y2` to `t2` indicate that the network's computed outputs (`y1`, `y2`) are intended to approximate or match specific target values (`t1`, `t2`). This comparison is crucial for calculating the error and subsequently training the network using algorithms like backpropagation.

In summary, this diagram precisely lays out the architecture of a 2-3-2 feedforward neural network, defining all its nodes and the specific notation for the weights that govern the flow of information through its layers. This structure is fundamental for understanding how such a network processes inputs to generate outputs and is the basis for algorithms like backpropagation used to train it.](images/1a6d08ea3053e54ff5e71bcf79c6bea7bb07a52776b725df216f51a48f4c23b7.jpg)
Figure 1: Backpropagation

We have two inputs: $x _ { 1 }$ and $x _ { 2 }$ . There is a single hidden layer with 3 units (nodes): $h _ { 1 }$ , $h _ { 2 }$ , and $h _ { 3 }$ . Finally, there are two outputs: $y _ { 1 }$ and $y _ { 2 }$ . The arrows that connect them are the weights. There are two weights matrices: w, and $\mathbf { u }$ . The w weights connect the input layer and the hidden layer. The $\mathbf { u }$ weights connect the hidden layer and the output layer. We have employed the letters $\mathbf { w }$ , and $\mathbf { u }$ , so it is easier to follow the computation to follow.

You can also see that we compare the outputs $y _ { 1 }$ and $y _ { 2 }$ with the targets $t _ { 1 }$ and $t _ { 2 }$ .

There is one last letter we need to introduce before we can get to the computations. Let $a$ be the linear combination prior to activation. Thus, we have: $\mathbf { a } ^ { ( 1 ) } = \mathbf { x w } + \mathbf { b } ^ { ( 1 ) }$ and $\mathbf { a } ^ { ( 2 ) } = \mathbf { h } \mathbf { u } + \mathbf { b } ^ { ( 2 ) }$ .

Since we cannot exhaust all activation functions and all loss functions, we will focus on two of the most common. A sigmoid activation and an L2-norm loss.

With this new information and the new notation, the output $y$ is equal to the activated linear combination. Therefore, for the output layer, we have $\mathbf { y } = \boldsymbol { \sigma } ( \mathbf { a } ^ { ( 2 ) } )$ , while for the hidden layer: $\mathbf { h } = \sigma ( \mathbf { a } ^ { ( 1 ) } )$ .

We will examine backpropagation for the output layer and the hidden layer separately, as the methodologies differ.

# 3 Useful formulas

I would like to remind you that:

$$
\operatorname { L 2 - n o r m } \operatorname { l o s s } \colon L = { \frac { 1 } { 2 } } \sum _ { i } ( y _ { i } - t _ { i } ) ^ { 2 }
$$

The sigmoid function is:

$$
\sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } }
$$

and its derivative is:

$$
\sigma ^ { \prime } ( x ) = \sigma ( x ) ( 1 - \sigma ( x ) )
$$

# 4 Backpropagation for the output layer

In order to obtain the update rule:

$$
\mathbf { u }  \mathbf { u } - \eta \nabla _ { \mathbf { u } } L ( \mathbf { u } )
$$

we must calculate

$$
\nabla _ { \mathbf { u } } L ( \mathbf { u } )
$$

Let’s take a single weight $u _ { i j }$ . The partial derivative of the loss w.r.t. $u _ { i j }$ equals:

$$
\frac { \partial L } { \partial u _ { i j } } = \frac { \partial L } { \partial y _ { j } } \frac { \partial y _ { j } } { \partial a _ { j } ^ { ( 2 ) } } \frac { \partial a _ { j } ^ { ( 2 ) } } { \partial u _ { i j } }
$$

where i corresponds to the previous layer (input layer for this transformation) and j corresponds to the next layer (output layer of the transformation). The partial derivatives were computed simply following the chain rule.

$$
{ \frac { \partial L } { \partial y _ { j } } } = ( y _ { j } - t _ { j } )
$$

following the L2-norm loss derivative.

$$
\frac { \partial y _ { j } } { \partial a _ { j } ^ { ( 2 ) } } = \sigma ( a _ { j } ^ { ( 2 ) } ) ( 1 - \sigma ( a _ { j } ^ { ( 2 ) } ) ) = y _ { j } ( 1 - y _ { j } )
$$

following the sigmoid derivative.

Finally, the third partial derivative is simply the derivative of $\mathbf { a } ^ { ( 2 ) } = \mathbf { h } \mathbf { u } + \mathbf { b } ^ { ( 2 ) }$ . So,

$$
\frac { \partial a _ { j } ^ { ( 2 ) } } { \partial u _ { i j } } = h _ { i }
$$

Replacing the partial derivatives in the expression above, we get:

$$
\frac { \partial L } { \partial u _ { i j } } = \frac { \partial L } { \partial y _ { j } } \frac { \partial y _ { j } } { \partial a _ { j } ^ { ( 2 ) } } \frac { \partial a _ { j } ^ { ( 2 ) } } { \partial u _ { i j } } = ( y _ { j } - t _ { j } ) y _ { j } ( 1 - y _ { j } ) h _ { i } = \delta _ { j } h _ { i }
$$

Therefore, the update rule for a single weight for the output layer is given by:

$$
u _ { i j }  u _ { i j } - \eta \delta _ { j } h _ { i }
$$

# 5 Backpropagation of a hidden layer

Similarly to the backpropagation of the output layer, the update rule for a single weight, $w _ { i j }$ would depend on:

$$
\frac { \partial L } { \partial w _ { i j } } = \frac { \partial L } { \partial h _ { j } } \frac { \partial h _ { j } } { \partial a _ { j } ^ { ( 1 ) } } \frac { \partial a _ { j } ^ { ( 1 ) } } { \partial w _ { i j } }
$$

following the chain rule.

Taking advantage of the results we have so far for transformation using the sigmoid activation and the linear model, we get:

$$
\frac { \partial h _ { j } } { \partial a _ { j } ^ { ( 1 ) } } = \sigma ( a _ { j } ^ { ( 1 ) } ) ( 1 - \sigma ( a _ { j } ^ { ( 1 ) } ) ) = h _ { j } ( 1 - h _ { j } )
$$

and

$$
\frac { \partial a _ { j } ^ { ( 1 ) } } { \partial w _ { i j } } = x _ { i }
$$

The actual problem for backpropagation comes from the term $\frac { \partial L } { \partial h _ { j } }$ That’s due to the fact that there is no ”hidden” target. You can follow the solution for weight $w _ { 1 1 }$ below. It is advisable to also check Figure 1, while going through the computations.

$$
\begin{array} { c } { { \displaystyle \frac { \partial { \cal L } } { \partial h _ { 1 } } = \frac { \partial { \cal L } } { \partial y _ { 1 } } \frac { \partial y _ { 1 } } { \partial a _ { 1 } ^ { ( 2 ) } } \frac { \partial a _ { 1 } ^ { ( 2 ) } } { \partial h _ { 1 } } + \frac { \partial { \cal L } } { \partial y _ { 2 } } \frac { \partial y _ { 2 } } { \partial a _ { 2 } ^ { ( 2 ) } } \frac { \partial a _ { 2 } ^ { ( 2 ) } } { \partial h _ { 1 } } = } } \\ { { = ( y _ { 1 } - t _ { 1 } ) y _ { 1 } ( 1 - y _ { 1 } ) u _ { 1 1 } + ( y _ { 2 } - t _ { 2 } ) y _ { 2 } ( 1 - y _ { 2 } ) u _ { 1 2 } } } \end{array}
$$

From here, we can calculate $\frac { \partial L } { \partial w _ { 1 1 } }$ which was what we wanted. The final expression is:

$$
\frac { \partial L } { \partial w _ { 1 1 } } = \left[ ( y _ { 1 } - t _ { 1 } ) y _ { 1 } ( 1 - y _ { 1 } ) u _ { 1 1 } + ( y _ { 2 } - t _ { 2 } ) y _ { 2 } ( 1 - y _ { 2 } ) u _ { 1 2 } \right] h _ { 1 } ( 1 - h _ { 1 } ) x _ { 1 }
$$

The generalized form of this equation is:

$$
\frac { \partial L } { \partial w _ { i j } } = \sum _ { k } ( y _ { k } - t _ { k } ) y _ { k } ( 1 - y _ { k } ) u _ { j k } h _ { j } ( 1 - h _ { j } ) x _ { i }
$$

# 6 Backpropagation generalization

Using the results for backpropagation for the output layer and the hidden layer, we can put them together in one formula, summarizing backpropagation, in the presence of L2-norm loss and sigmoid activations.

$$
\frac { \partial L } { \partial w _ { i j } } = \delta _ { j } x _ { i }
$$

where for a hidden layer

$$
\delta _ { j } = \sum _ { k } \delta _ { k } w _ { j k } y _ { j } ( 1 - y _ { j } )
$$

Kudos to those of you who got to the end.

Thanks for reading.