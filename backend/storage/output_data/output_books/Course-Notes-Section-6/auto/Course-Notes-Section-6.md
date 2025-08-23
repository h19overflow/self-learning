# 365VDataScience

# MACHINE LEARNING

COURSE NOTES – SECTION 6

# IT’S TIME TO DIG DEEPER

An initial linear combination and the added non-linearity form a layer. The layer is the building block of neural networks.

# Minimal example (a simple neural network)

# Neural networks

![## Image Analysis: 1517e3f8f96e99d742875dd593c8235d4422e41dbbe1a7d4eb659a21f65484dc.jpg

**Conceptual Understanding:**
This image conceptually represents the most basic feedforward neural network structure. Its main purpose is to illustrate the fundamental components and the directional flow of information within such a network. Key ideas conveyed are the concepts of an 'Input layer' responsible for receiving data and an 'Output layer' responsible for producing results, with individual nodes (x₁, x₂, y) representing data points or computational units within these layers.

**Content Interpretation:**
The image illustrates the fundamental architecture of a simple neural network. It shows two input nodes (x₁ and x₂) within an 'Input layer' that feed into a single output node (y) within an 'Output layer'. This represents a direct mapping or transformation of two input features into one output. The arrows signify the direction of data flow and computation, where inputs are processed to yield an output. This is a foundational concept demonstrating how data can be received, processed through a minimal structure, and then an outcome generated.

**Key Insights:**
The main takeaway from this image is the clear distinction and connection between an 'Input layer' and an 'Output layer' in a neural network. It highlights that multiple inputs (x₁, x₂) can contribute to a single output (y). The visual representation emphasizes the concept of directed information flow, where input data is fed into a system and transformed into a desired output. This simple model demonstrates the core principle of how neural networks process information.

**Document Context:**
Given the document context 'Neural networks,' this image serves as an introductory or foundational diagram. It visually explains the most basic structure of a neural network, consisting of an input and an output layer. This simple representation is crucial for understanding more complex neural network architectures by breaking down the fundamental components and information flow, laying the groundwork for discussions on nodes, layers, and data processing within the network.

**Summary:**
This image displays a basic representation of a neural network, illustrating the flow of information from an Input layer to an Output layer. The Input layer, shaded in light pink, contains two distinct input nodes, labeled 'x₁' and 'x₂'. These nodes represent individual input features or data points. Directed arrows originate from both 'x₁' and 'x₂', converging towards a single node in the Output layer. The Output layer, also shaded in light pink, contains one output node, labeled 'y'. This 'y' node represents the computed output or result derived from the inputs. The arrows clearly indicate a unidirectional flow, signifying that information from the input nodes 'x₁' and 'x₂' is processed and combined to produce the output 'y'. The diagram visually separates the input and output functionalities into distinct layers.](images/1517e3f8f96e99d742875dd593c8235d4422e41dbbe1a7d4eb659a21f65484dc.jpg)

![## Image Analysis: d4f922adae1da83cebae0de5083a16830bbdf187d3cb5c717fbe94214037686e.jpg

**Conceptual Understanding:**
This image conceptually represents a simplified model of a single artificial neuron or a computational unit within a neural network. Its main purpose is to clearly illustrate the two primary mathematical operations involved in a neural network node's computation: first, a linear transformation (a weighted sum of inputs plus a bias), and second, the application of a non-linear activation function. The key ideas communicated are the sequential flow from inputs through these transformations to generate an output, emphasizing the critical role of non-linearity in enabling complex pattern learning.

**Content Interpretation:**
The image depicts the core computational steps of a single neuron or node in a neural network. It shows the intake of multiple inputs ("x₁" and "x₂") from the "Input layer". These inputs are then subjected to a "Linear combination", represented by the formula "xw+b", which signifies a weighted sum of inputs plus a bias. Subsequently, the result of this linear combination passes through a "Non-linearity" stage, indicated by a square box containing a curved line symbol, which represents an activation function. This step introduces non-linear behavior, crucial for the network's ability to model complex data patterns. Finally, the processed signal emerges as the "Output", denoted by "y". The sequential flow highlights how raw inputs are transformed through these mathematical operations into a meaningful output.

**Key Insights:**
The image teaches several key lessons: 
1.  **Fundamental Neuron Structure:** It illustrates the basic architecture and computational process of an artificial neuron, the core building block of neural networks.
2.  **Two-Step Transformation:** A neuron processes inputs in two sequential stages: a "Linear combination" (weighted sum and bias) and then a "Non-linearity" (activation function).
3.  **Importance of Non-linearity:** The explicit "Non-linearity" stage is critical; it enables neural networks to model complex, non-linear relationships in data, distinguishing them from simpler linear models. Without it, stacking multiple layers would still only yield a linear output. 

These insights are directly supported by the textual elements: "Input layer", "x₁", "x₂" show the starting point; "Linear combination", "xw+b" detail the linear operation; "Non-linearity" and its symbol highlight the crucial non-linear transformation; and "Output", "y" represent the final result.

**Document Context:**
Given the document section "Neural networks," this image serves as a foundational diagram to explain the basic computational unit of a neural network – a single neuron or node. It is likely introduced early to help readers understand the fundamental operations before discussing more complex architectures, training, or applications. It directly addresses how a neuron processes information from its inputs to produce an output, laying the groundwork for understanding the 'how' of neural network functionality.

**Summary:**
This diagram illustrates the fundamental computational process within a single artificial neuron, a basic building block of neural networks. The process flows from left to right, transforming initial inputs into a final output through a series of well-defined steps.

**Step-by-Step Process:**

1.  **Input Layer:** The process begins with the "Input layer," which receives initial data or signals. In this diagram, two distinct inputs are shown: "x₁" and "x₂". These represent the raw features or values that the neuron will process.

2.  **Linear Combination:** Next, these inputs (x₁ and x₂) are directed to a "Linear combination" step. Here, the inputs are mathematically combined. The text within the circular node for this stage, "xw+b", explicitly shows this operation: it involves multiplying each input (x) by its corresponding weight (w) and summing these products, then adding a bias term (b). This step produces a single value that is a linear transformation of the inputs.

3.  **Non-linearity:** The result from the "Linear combination" (xw+b) is then fed into the "Non-linearity" stage. This is represented by a square box containing a curved line symbol, which is a common visual shorthand for an activation function. The purpose of this step is to introduce non-linearity into the model. Without non-linearity, a neural network, no matter how deep, would only be capable of performing linear transformations. The activation function allows the neuron to learn and represent more complex, non-linear relationships in the data.

4.  **Output:** Finally, the output of the "Non-linearity" stage becomes the neuron's "Output," denoted by "y" within a circular node. This "y" is the final processed value or prediction generated by this single computational unit, which can then be passed on as an input to other neurons in a deeper network, or directly interpreted as the final result.

In essence, this diagram visually explains how a neuron takes raw inputs, performs a linear mathematical operation on them, applies a non-linear transformation, and then produces an output, forming the core mechanism by which neural networks learn from and make predictions on data.](images/d4f922adae1da83cebae0de5083a16830bbdf187d3cb5c717fbe94214037686e.jpg)

In the minimal example we trained a neural network which had no depth. There were solely an input layer and an output layer. Moreover, the output was simply a linear combination of the input.

Neural networks step on linear combinations, but add a nonlinearity to each one of them. Mixing linear combinations and non-linearities allows us to model arbitrary functions.

This is a deep neural network (deep net) with 5 layers.

Input layer

Hidden layer 1 Hidden layer 2 Hidden layer 3

How to read this diagram:

A layer

A unit (a neuron)

# Output laye X X V 双 双 KX X XX M A XX AT X N 1

Arrows represent mathematical transformations

Hidden layer 1 Hidden layer 2 Hidden layer 3

The width of a layer is the number of units in that layer

The width of the net is the number of units of the biggest layer

The depth of the net is equal to the number of layers or the number of hidden layers. The term has different definitions. More often than not, we are interested in the number of hidden layers (as there are always input and output layers).

# Width

# put layer Output layer 双 双 大 ? Depth

The width and the depth of the net are called hyperparameters. They are values we manually chose when creating the net.

# Why we need non-linearities to stack layers

You can see a net with no non-linearities: just linear combinations.

Two consecutive linear transformations are equivalent to a single one.

![## Image Analysis: d45be95cef588137fb90d3bde45e4912b3e426137d9b0447dd760e695f59e57f.jpg

**Conceptual Understanding:**
This image conceptually represents the limitation of purely linear transformations in a multi-layered neural network. The main purpose is to demonstrate that stacking linear layers is mathematically equivalent to having only a single linear layer. The key idea being communicated is that depth (multiple layers) in a neural network is redundant if only linear operations are performed, thus highlighting the critical role of non-linearities for a neural network to learn complex patterns and benefit from its layered structure.

**Content Interpretation:**
The image illustrates the architecture of a simple feedforward neural network and, more critically, provides a mathematical demonstration of why stacking multiple linear layers is equivalent to a single linear transformation. The neural network diagram shows an input layer, a hidden layer, and an output layer, connected by weights. The equations translate these connections into matrix multiplications. The core concept is the algebraic simplification of sequential matrix multiplications (`h = x * w₁` followed by `y = h * w₂`) into a single matrix multiplication (`y = x * W*`), where `W*` is the product of `w₁` and `w₂`. This process demonstrates that the 'depth' provided by a hidden layer in a purely linear network does not offer additional computational complexity or learning capacity.

**Key Insights:**
The main takeaway from this image is that applying two or more consecutive linear transformations (like matrix multiplications in a neural network) results in a single, equivalent linear transformation. This means that a deep neural network composed solely of linear layers does not gain any additional expressive power compared to a shallow network with just one linear layer. The specific textual evidence like "h = x * w₁", "y = h * w₂", and their simplification to "y = x * W*" with the final statement "Two consecutive linear transformations are equivalent to a single one." explicitly supports this insight. Therefore, to model complex, non-linear relationships and benefit from the depth of a multi-layered architecture, non-linear activation functions are essential between the linear layers.

**Document Context:**
This image is crucial for the document's section titled "Why we need non-linearities to stack layers." It provides the fundamental mathematical proof for the necessity of non-linear activation functions in deep learning. Without the non-linearities, as demonstrated by this image, a multi-layered neural network would simply collapse into an equivalent single-layer model, unable to learn complex, non-linear patterns that are characteristic of most real-world data. The image serves as a visual and mathematical foundation, explaining why the architectural depth of neural networks is only beneficial when non-linear transformations are introduced between layers.

**Summary:**
The image displays a conceptual diagram of a feedforward neural network alongside mathematical equations demonstrating the effect of stacking linear layers. On the left, a neural network is depicted with three distinct layers: an "Input layer" on the far left, a "Hidden layer" in the middle, and an "Output layer" on the far right. The Input layer consists of 8 nodes (circles), the Hidden layer has 9 nodes, and the Output layer has 4 nodes. All nodes between adjacent layers are fully connected by lines, representing weights. Below the neural network diagram, red arrows point upwards to the respective layers, with a corresponding mathematical expression: "x * w₁ = h * w₂ = y". Beneath this expression, the dimensions of these matrices/vectors are specified: "1x8" for `x`, "8x9" for `w₁`, "1x9" for `h`, "9x4" for `w₂`, and "1x4" for `y`. This shows the input `x` transforming via weight matrix `w₁` to `h`, and `h` transforming via `w₂` to the output `y` in a sequential manner.

On the right side of the image, a series of equations illustrates a mathematical simplification. The first equation is "h = x * w₁". The second is "y = h * w₂". A red arrow points from the second equation to the next step, which is "y = x * W₁ * W₂". This step combines the previous two, substituting `h` with `x * w₁`. The terms "W₁ * W₂" are enclosed in a box, and their individual dimensions are noted below them as "8x9" for `W₁` and "9x4" for `W₂`. Another red arrow points from this combined expression to the final simplified form: "y = x * W*", where `W` has a star superscript. Below `W*`, its dimension is given as "8x4". A large curly bracket encompasses all the equations and a descriptive sentence on the right. The sentence reads: "Two consecutive linear transformations are equivalent to a single one."

In essence, the image visually and mathematically demonstrates that if a neural network uses only linear transformations (multiplication by weight matrices) between its layers, stacking multiple layers (like the input, hidden, and output layers shown) does not increase its representational power beyond that of a single linear transformation. The initial input `x` multiplied by `w₁` yields `h`, which is then multiplied by `w₂` to yield `y`. Mathematically, this is equivalent to `x` being multiplied by a single combined weight matrix `W*`, which is the product of `W₁` and `W₂`. This `W*` matrix effectively performs the entire transformation in one step, making the intermediate hidden layer redundant in terms of learning capacity for purely linear operations.](images/d45be95cef588137fb90d3bde45e4912b3e426137d9b0447dd760e695f59e57f.jpg)

# Input

![## Image Analysis: 3193d105de722342850676737ce63b75c71ccd835b30c4ed89946c6148429f57.jpg

**Conceptual Understanding:**
This image conceptually represents a simplified model of an artificial neuron or a single processing unit within an artificial neural network. The main purpose of this diagram is to illustrate the fundamental computational steps involved in how such a unit processes incoming data and generates an output. It clearly communicates the key ideas of input aggregation (through linear combination) and subsequent non-linear transformation (via an activation function).

**Content Interpretation:**
This image displays a fundamental process flow for an artificial neuron or a processing unit in a neural network. It illustrates the sequence of operations from receiving inputs to producing an output. The processes shown are:
1.  **Input Reception:** Two empty circles on the far left represent initial inputs into the system.
2.  **Linear Combination:** The middle empty circle, explicitly labeled "Linear combination," depicts the aggregation of these inputs. In neural networks, this typically involves a weighted sum of the inputs.
3.  **Activation Function:** The rectangular box containing a squiggly line, labeled "Activation function," represents the application of a non-linear function to the result of the linear combination. This function introduces non-linearity, which is crucial for the network's ability to learn complex patterns.
4.  **Output Generation:** The final arrow extending to the right indicates the output of this processing unit.

The significance of this sequence is that it forms the basic building block of artificial neural networks. The "Linear combination" step gathers and consolidates information from various sources, while the "Activation function" step then transforms this aggregated information in a non-linear way, allowing the model to learn and represent intricate relationships within data. All extracted text elements directly label and describe these sequential operations, providing explicit evidence for the interpretation of each stage.

**Key Insights:**
The main takeaway from this image is that an artificial neuron or a basic processing unit in machine learning processes inputs in two distinct and sequential stages: first, a linear aggregation, and second, a non-linear transformation. The specific text elements "Linear combination" and "Activation function" provide direct evidence for these two critical stages.

Key insights supported by this image and its textual elements include:
*   **Modularity of Neural Networks:** This diagram illustrates the fundamental computational unit from which larger and more complex neural networks are built.
*   **Importance of Linear Combination:** The "Linear combination" step highlights that inputs are first aggregated, typically through weighted sums, before further processing. This aggregation step consolidates information from multiple sources.
*   **Crucial Role of Activation Functions:** The "Activation function" step emphasizes that a non-linear transformation is applied to the aggregated input. This non-linearity is vital for neural networks to learn and model non-linear relationships in data, allowing them to solve complex real-world problems that linear models cannot address. Without activation functions, a neural network, no matter how many layers it has, would behave like a single linear model.

**Document Context:**
Within a document on machine learning, deep learning, or neural networks, this image likely serves as an introductory diagram explaining the fundamental operation of a single perceptron or artificial neuron. It sets the foundation for understanding how more complex neural network architectures are constructed and how they process information. Given the section title "Input," this diagram specifically explains how initial inputs are processed at the most basic unit level, detailing the transformation steps before further processing or output.

**Summary:**
This diagram illustrates the core computational steps within a single processing unit, often referred to as a neuron, in an artificial neural network. The process begins with multiple inputs, represented by the two empty circles on the far left. These inputs are then directed via arrows to a central empty circle, which is explicitly labeled "Linear combination." This step signifies that the inputs are combined arithmetically, typically by being multiplied by weights and then summed up. The output of this "Linear combination" is then fed forward to a rectangular box containing a squiggly line symbol. This box is labeled "Activation function," indicating that a non-linear mathematical function is applied to the result of the linear combination. This activation function determines the final output of the processing unit, which is then represented by the arrow extending to the right. In essence, this diagram shows how raw inputs are transformed first linearly and then non-linearly to produce an output, a foundational concept for understanding how neural networks learn and make predictions.](images/3193d105de722342850676737ce63b75c71ccd835b30c4ed89946c6148429f57.jpg)

In the respective lesson, we gave an example of temperature change. The temperature starts decreasing (which is a numerical change). Our brain is a kind of an ‘activation function’. It tells us whether it is cold enough for us to put on a jacket.

Putting on a jacket is a binary action: 0 (no jacket) or 1 ( jacket).

This is a very intuitive and visual (yet not so practical) example of how activation functions work.

Activation functions (non-linearities) are needed so we can break the linearity and represent more complicated relationships.

Moreover, activation functions are required in order to stack layers.

Activation functions transform inputs into outputs of a different kind.

![## Image Analysis: 2b4ac723b27ef8b3d6eab16a7f8e0c29b67cb1380a365dcc5935b7233ae05755.jpg

**Conceptual Understanding:**
This image represents the conceptual flow of an input signal through a processing unit, specifically an 'ACTIVATION FUNCTION'. Its main purpose is to illustrate that an activation function can transform an input into two fundamentally different types of outputs: 'Linear' and 'Non-linear'. The key ideas communicated are the input-processing-output pipeline and the distinct nature of linear versus non-linear transformations, particularly relevant in fields like artificial intelligence and machine learning where activation functions are integral components of neural networks.

**Content Interpretation:**
The image conceptually illustrates the role of an 'ACTIVATION FUNCTION' in processing an 'INPUT' to generate an 'OUTPUT'. It demonstrates that this function can produce two distinct types of outputs: 'Linear' and 'Non-linear'. The thermometer as 'INPUT' suggests a continuous numerical value. The brain symbol for 'ACTIVATION FUNCTION' represents the computational or transformative step, analogous to a neuron processing a signal. The two different human figures as 'OUTPUT' symbolize the distinct outcomes or representations of these two types of processing. The 'Linear' output path implies a direct, proportional relationship to the input, while the 'Non-linear' path indicates a more complex, often required for sophisticated pattern recognition and learning, transformation.

**Key Insights:**
The main takeaway from this image is that an 'ACTIVATION FUNCTION' is a crucial component that takes an 'INPUT' and transforms it into an 'OUTPUT'. The image specifically highlights that an activation function can yield both 'Linear' and 'Non-linear' outputs. The distinction between 'Linear' and 'Non-linear' is fundamental, as non-linearity is essential for neural networks to learn and model complex, real-world data patterns that linear models cannot capture. The specific text labels 'INPUT', 'ACTIVATION FUNCTION', 'OUTPUT', 'Linear', and 'Non-linear' are critical evidence, clearly defining each stage and the possible outcomes of the processing.

**Document Context:**
Given the document context 'Section: Input', this image serves as a foundational explanation of how inputs are handled and transformed within a system, likely an artificial neural network. It introduces the critical component of an 'ACTIVATION FUNCTION' and its capacity to produce different output types (linear vs. non-linear), which directly impacts the system's ability to learn and model complex relationships. This sets the stage for understanding the subsequent processing of input data and the behavior of the overall system.

**Summary:**
This image illustrates the fundamental concept of an 'ACTIVATION FUNCTION' within a system, demonstrating how an 'INPUT' can be processed to yield different types of 'OUTPUT' – specifically 'Linear' and 'Non-linear'. The process begins with an 'INPUT', visually represented by a thermometer showing a red liquid column, indicating a measurable value. This 'INPUT' flows via an arrow into the 'ACTIVATION FUNCTION', symbolized by a pink brain icon, which signifies the processing or transformation unit. From the 'ACTIVATION FUNCTION', two distinct paths emerge, each leading to a different 'OUTPUT'. The upper path is labeled 'Linear' and points to a male figure dressed in a suit and tie, standing upright with a neutral expression. The lower path is labeled 'Non-linear' and points to a male figure in a white shirt and blue tie, with rolled-up sleeves and hands together, suggesting a more dynamic or complex posture. Both paths are indicated by arrows originating from the 'ACTIVATION FUNCTION' and culminating at their respective 'OUTPUT' figures. The overall diagram provides a clear visual metaphor for how an input signal undergoes an activation process, resulting in either a linear or a non-linear output, a core concept in artificial neural networks.](images/2b4ac723b27ef8b3d6eab16a7f8e0c29b67cb1380a365dcc5935b7233ae05755.jpg)

# Common activation functions

<table><tr><td>Name</td><td>Formula</td><td>Derivative</td><td>Graph</td><td>Range</td></tr><tr><td>sigmoid (logistic function)</td><td>σ(a) = 1+-a</td><td>σ(@=σ(@（1-g(@））</td><td>1 0.5- 0 0</td><td>(0,1)</td></tr><tr><td>(hyperbTain tangent)</td><td>e-e-a tanh(@)=a+e-a</td><td>tan@=+-a)</td><td>1 01 0 £ -1</td><td> (-1,1)</td></tr><tr><td>(rectified inea unt)</td><td> relu(a) = max(0,a)</td><td>ro@_{.if≤0</td><td>▲</td><td>（0,8）</td></tr><tr><td> softmax</td><td>g(@）=</td><td></td><td>0 0</td><td>(0,1)</td></tr></table>

All common activation functions are: monotonic, continuous, and differentiable. These are important properties needed for the optimization.

# Input layer

![## Image Analysis: afae07826051cb4bf429bf6452d0500a9511473f7bf8d04932fa3a85a113d5f3.jpg

**Conceptual Understanding:**
This image conceptually represents a simplified feed-forward artificial neural network used for image classification. Its main purpose is to illustrate how an input image is processed through different computational layers (hidden and output) to generate a classified output with associated probabilities. The image communicates the key ideas of layered network architecture, the flow of data, the internal calculations within neurons (suggested by a_h = hw+b), and the final output as a probability distribution over predefined classes, leading to a specific classification decision (identifying the horse).

**Content Interpretation:**
This image illustrates a simplified feed-forward artificial neural network engaged in an image classification task. It depicts the processing of an input image (a horse) through a 'Hidden layer' and an 'Output layer' to produce a probabilistic classification. The 'Hidden layer' nodes ('h1', 'h2', 'h3') represent intermediate computational stages where features from the input are extracted and transformed. The 'Output layer' nodes ('0.1', '0.2', '0.7') provide the network's confidence scores or probabilities for different classes, 'cat', 'dog', and 'horse', respectively. The connections between layers (represented by lines with arrows) signify the flow of information and the weighted relationships between neurons. The equation 'a_h = hw+b' is a fundamental component of neural network computations, representing the weighted sum of inputs ('hw') plus a bias ('b'), which is then often passed through an activation function to produce the output 'a_h' for a given neuron.

**Key Insights:**
The image effectively teaches several key concepts of neural networks and image classification: 1. Neural Network Architecture: It shows a basic three-layer structure (input, hidden, output) for processing information. 2. Layered Processing: Information flows sequentially from the input through the 'Hidden layer' (h1, h2, h3) to the 'Output layer' (0.1, 0.2, 0.7). 3. Probabilistic Classification: The 'Output layer' provides numerical scores (0.1 for 'cat', 0.2 for 'dog', 0.7 for 'horse') that can be interpreted as probabilities or confidence levels for each potential class. 4. Prediction Mechanism: The class with the highest output value ('0.7' for 'horse') represents the network's prediction, which aligns with the input image. 5. Core Neuron Calculation: The equation 'a_h = hw+b' highlights the fundamental linear transformation (weighted sum plus bias) that occurs within each neuron before activation, providing insight into the mathematical operations. These insights are directly evidenced by the explicit labels 'Hidden layer', 'Output layer', the numerical values and class labels in the output, and the mathematical formula provided.

**Document Context:**
Given the document section 'Input layer', this image serves as a direct visual example of how an input, in this case, an image of a horse, is introduced into a neural network system. It demonstrates the initial step of information flow from the raw input into the computational layers. The image provides a concrete illustration of the concept that an input is processed, transformed through hidden computations, and eventually leads to a specific output classification. It visually explains the transition from an unclassified input to a categorized output through the defined layers and the underlying mathematical operations.

**Summary:**
The image displays an example of an artificial neural network performing image classification. On the left, there is an input image showing the head of a horse, characterized by its brown and white markings, light brown mane, and a white blaze on its face, set against a blurry green and blue background. This image serves as the input to the neural network depicted on the right. The neural network consists of three main conceptual parts: an implicit input layer (represented by the horse image), a 'Hidden layer', and an 'Output layer'. The 'Hidden layer' contains three nodes labeled 'h1', 'h2', and 'h3'. Each of these hidden nodes receives input from the preceding layer (the input image) via multiple connections, indicated by converging arrows. From the 'Hidden layer', connections radiate to each node in the 'Output layer'. The 'Output layer' contains three nodes with numerical values: '0.1', '0.2', and '0.7'. Each of these output nodes is associated with a specific classification label: '0.1' points to 'cat', '0.2' points to 'dog', and '0.7' points to 'horse'. Below the network diagram, a mathematical equation is displayed: 'a_h = hw+b'. This equation likely represents the activation function or weighted sum calculation performed within the neurons of the network. The overall representation demonstrates how an input image is processed through layers of a neural network to produce a classification output, with the highest probability (0.7) correctly identifying the input as a 'horse'.](images/afae07826051cb4bf429bf6452d0500a9511473f7bf8d04932fa3a85a113d5f3.jpg)

The softmax activation transforms a bunch of arbitrarily large or small numbers into a valid probability distribution.

While other activation functions get an input value and transform it, regardless of the other elements, the softmax considers the information about the whole set of numbers we have.

The values that softmax outputs are in the range from 0 to 1 and their sum is exactly 1 (like probabilities).

# Example:

$\pmb { \mathrm { a } } = [ - 0 . 2 1 , 0 . 4 7 , 1 . 7 2 ]$   
$\mathsf { S o f t m a x } \left( \mathsf { a } \right) = \frac { e ^ { a _ { i } } } { \sum _ { j } e ^ { a _ { j } } }$   
$\textstyle \sum _ { j } e ^ { a _ { j } } = e ^ { - 0 . 2 1 } + e ^ { 0 . 4 7 } + e ^ { 1 . 7 2 } = 8$   
$\mathsf { s o f t m a x } \left( \mathsf { a } \right) = \bigl [ \frac { e ^ { - 0 . 2 1 } } { 8 } , \frac { e ^ { 0 . 4 7 } } { 8 } , \frac { e ^ { 1 . 7 2 } } { 8 } \bigr ]$   
$y = [ 0 . 1 , 0 . 2 , 0 . 7 ] $ probability distribution

The property of the softmax to output probabilities is so useful and intuitive that it is often used as the activation function for the final (output) layer.

However, when the softmax is used prior to that (as the activation of a hidden layer), the results are not as satisfactory. That’s because a lot of the information about the variability of the data is lost.

![## Image Analysis: ff1036e239b63670b30a96faeafed6c62a6840ff5d26c48dcc8d84715f3fd305.jpg

**Conceptual Understanding:**
This image conceptually represents the fundamental operations within a simple feedforward neural network, illustrating both the forward pass for generating predictions and the backward pass for learning through error correction (backpropagation). Its main purpose is to visually explain the flow of data, the role of connection weights, and the mechanism by which errors are calculated and used to update the network's parameters during the training phase. The two distinct diagrams, using different arrow colors, effectively convey the directional nature of these two critical processes in machine learning.

**Content Interpretation:**
The image illustrates the fundamental architecture and operational principles of a feedforward neural network, specifically highlighting the forward propagation of data and the backward propagation of errors. The top diagram (green arrows) shows the 'forward pass' where input data (x1, x2) is processed through weighted connections (w11-w23) to a hidden layer (h1, h2, h3), then further processed through another set of weighted connections (u11-u32) to produce outputs (y1, y2). The error terms (e1, e2) are derived by comparing the network outputs (y1, y2) with the target values (t1, t2). The bottom diagram (red arrows) shows the 'backward pass' or backpropagation, where these errors (e1, e2) are propagated backward through the network, adjusting the weights (u11-u32 and w11-w23) in a reverse direction to minimize the discrepancy between predicted and actual outputs. This bidirectional visualization of data flow and error correction is central to understanding how neural networks learn. The specific textual labels for nodes (x, h, y), weights (w, u), errors (e), and targets (t), along with their subscripts, provide precise identification of each component and its role in the network's calculations.

**Key Insights:**
The image provides several key pieces of knowledge: 1. **Neural Network Architecture:** It depicts a three-layer feedforward neural network (input, hidden, output) with two input nodes (x1, x2), three hidden nodes (h1, h2, h3), and two output nodes (y1, y2). 2. **Weighted Connections:** The learning process in the network is governed by weights (w and u) that modulate the strength of connections between neurons. These weights are explicitly labeled with subscripts indicating their source and destination, e.g., 'w11' connects x1 to h1. 3. **Forward Pass:** Information flows from input to output (green arrows), where inputs are processed through hidden layers to produce predictions. This path is represented by: Inputs (x1, x2) -> Weighted connections (w11-w23) -> Hidden Layer (h1, h2, h3) -> Weighted connections (u11-u32) -> Output Layer (y1, y2). 4. **Error Calculation:** Outputs (y1, y2) are compared with target values (t1, t2) to calculate errors (e1, e2). 5. **Backward Pass (Backpropagation):** Errors are propagated backward through the network (red arrows) to adjust the connection weights (u and w). This process involves: Errors (e1, e2) -> Backwards through weights (u11-u32) -> Hidden Layer (h1, h2, h3) -> Backwards through weights (w11-w23) -> Input Layer (x1, x2). 6. **Bidirectional Flow:** The distinct green and red arrows clearly differentiate the forward propagation of data from the backward propagation of errors, visually explaining the iterative learning process in neural networks.

**Document Context:**
This image serves as a foundational visual aid for explaining the core mechanism of neural networks, particularly in the context of supervised learning. It directly illustrates the two key phases: the forward computation of outputs and the backward adjustment of weights, which are essential for training the network. In a document, this image would likely appear in a section introducing neural network architecture, explaining backpropagation, or discussing the mathematical operations involved in machine learning models. It provides the visual and textual details necessary for readers to grasp how data flows, how intermediate calculations are performed (implied by weights), and how errors are used to update the model, thus supporting the understanding of concepts like model training and optimization.

**Summary:**
This image displays two diagrams of a feedforward neural network, illustrating the forward pass (top diagram with green arrows) and the backward pass (bottom diagram with red arrows) through the network layers. Both diagrams show a network with an input layer (x1, x2), a hidden layer (h1, h2, h3), and an output layer (y1, y2). The connections between layers are weighted, denoted by 'w' for input-to-hidden connections and 'u' for hidden-to-output connections, each with specific subscripts indicating the source and destination nodes. The top diagram depicts the flow of information from input to output, while the bottom diagram shows the propagation of error backward from the output layer towards the input layer, with the error (e) and target (t) values associated with the outputs. The clear distinction in arrow color helps visually differentiate the two processes.](images/ff1036e239b63670b30a96faeafed6c62a6840ff5d26c48dcc8d84715f3fd305.jpg)

Forward propagation is the process of pushing inputs through the net. At the end of each epoch, the obtained outputs are compared to targets to form the errors.

Backpropagation of errors is an algorithm for neural networks using gradient descent. It consists of calculating the contribution of each parameter to the errors. We backpropagate the errors through the net and update the parameters (weights and biases) accordingly.

![## Image Analysis: adfd4c6cc847d1e5d60528724af78fa78af286c1cbffd1372d9c0e8d8812ffb9.jpg

**Conceptual Understanding:**
This image conceptually represents a **multi-layer feedforward neural network**, a fundamental architecture in artificial intelligence and machine learning. Its main purpose is to illustrate the flow of information through such a network, from input to output, and to highlight key components like neurons, synaptic weights, and activation functions. Furthermore, it explicitly includes elements that suggest the process of **error backpropagation**, a mechanism used for training neural networks by adjusting weights based on the discrepancy between predicted and actual outputs. The key ideas communicated are the hierarchical structure of a neural network, the role of weighted connections in transforming data, the application of activation functions, and the concept of error calculation and its backward propagation for learning.

**Content Interpretation:**
The image depicts a specific instance of a neural network model, showcasing the following processes, concepts, and relationships:

*   **Input Layer and Input Data (x1, x2):** The circles labeled 'x1' and 'x2' represent the input layer neurons. These are where the raw data or features are fed into the network. The presence of two input nodes indicates that the network processes two distinct input features.
    *   *Textual Evidence:* "x1", "x2".

*   **Hidden Layer and Feature Transformation (h1, h2, h3):** The circles labeled 'h1', 'h2', and 'h3' represent the neurons in a single hidden layer. This layer performs non-linear transformations on the input data to learn increasingly abstract representations or features.
    *   *Textual Evidence:* "h1", "h2", "h3".

*   **Synaptic Weights (w_ij, u_ij):** The lines connecting neurons between layers are labeled with 'w' and 'u' symbols, representing synaptic weights. These weights quantify the strength of the connection between neurons and are the primary parameters learned during training.
    *   `w_ij` (e.g., w₁₁, w₁₂, w₁₃, w₂₁, w₂₂, w₂₃): These weights connect the input layer neurons (x_i) to the hidden layer neurons (h_j). For example, `w₁₁` is the weight connecting input `x1` to hidden neuron `h1`.
    *   `u_ij` (e.g., u₁₁, u₁₂, u₂₁, u₂₂, u₃₁, u₃₂): These weights connect the hidden layer neurons (h_i) to the output layer neurons (y_j). For example, `u₁₁` is the weight connecting hidden neuron `h1` to output neuron `y1`.
    *   *Textual Evidence:* "w₁₁", "w₁₂", "w₁₃", "w₂₁", "w₂₂", "w₂₃", "u₁₁", "u₁₂", "u₂₁", "u₂₂", "u₃₁", "u₃₂".

*   **Activation Functions (Red 'S' curve in a red square):** The red square containing an 'S' curve symbol typically denotes a non-linear activation function (like a sigmoid function). These functions introduce non-linearity into the network, enabling it to learn complex patterns and relationships in data.
    *   The placement of these activation functions with red arrows suggests their application at specific points:
        *   One on the path from `x1` to `h1` (associated with `w₁₁`) with a red arrow to `h1`. This could be highlighting the activation involved in computing `h1`'s input or an unusual feedback.
        *   Two on the paths leading to the output nodes `y1` and `y2` (from `h1`'s connection line with `u₁₁` and `u₁₂` respectively), with red arrows pointing to `y1` and `y2`. This indicates that an activation function is applied to the weighted sum before producing the final output values.
    *   *Textual Evidence:* The graphical 'S' curve symbol within a red square on the connection lines.

*   **Output Layer and Predictions (y1, y2):** The circles labeled 'y1' and 'y2' represent the output layer neurons, which produce the final predictions or classifications of the network.
    *   *Textual Evidence:* "y1", "y2".

*   **Target Values (t1, t2):** The labels 't1' and 't2' represent the desired or true target values that the network is trying to predict. The blue arrows from y1 to t1 and y2 to t2 show the comparison between the network's output and the actual targets.
    *   *Textual Evidence:* "t1", "t2".

*   **Error Signals (e1, e2):** The incoming red arrows labeled 'e1' and 'e2' pointing to 'y1' and 'y2' respectively, represent the error (difference between predicted output and target output) at the output layer. These errors are crucial for the learning process.
    *   *Textual Evidence:* "e1", "e2".

*   **Error Backpropagation (Red Feedback Arrows):** The thick red arrows originating from 'y1' and 'y2' and pointing backward to 'h1' illustrate the concept of error backpropagation. This is a mechanism where the error calculated at the output layer is propagated backward through the network to update the weights in the preceding layers. Specifically, the error from both `y1` and `y2` is shown feeding back to `h1`, suggesting `h1`'s weights are adjusted based on both output errors.
    *   *Textual Evidence:* Red arrows from y1 to h1 and y2 to h1.

*   **Specific Weight Highlight (Red arrow near w₁₁):** The leftmost red arrow, pointing leftwards from the vicinity of `w₁₁`, is a specific highlight. It might emphasize the starting point of a weight update process or draw attention to the specific weight `w₁₁` in a broader context not fully detailed by the diagram.
    *   *Textual Evidence:* Red arrow near "w₁₁".

The significance of this diagram is to visually represent the computational graph of a simple neural network and the direction of information flow during both the forward pass (prediction) and a partial backward pass (error propagation for learning). The interplay of inputs, weighted sums, activation functions, and error feedback is central to how neural networks learn.

**Key Insights:**
The image provides several key takeaways and insights into the functioning and structure of a basic neural network:

*   **Hierarchical, Layered Structure:** Neural networks process information through distinct layers (input, hidden, output). Each layer transforms the data before passing it to the next.
    *   *Textual Evidence:* "x1", "x2" (input layer); "h1", "h2", "h3" (hidden layer); "y1", "y2" (output layer).

*   **Weight-Driven Connections:** The strength of connections between neurons is governed by adjustable weights. These weights are the primary parameters learned during the training process, enabling the network to map inputs to desired outputs.
    *   *Textual Evidence:* All "w" and "u" labels (e.g., "w₁₁", "u₃₂").

*   **Non-Linearity via Activation Functions:** Activation functions introduce non-linearity, which is crucial for the network to learn complex, non-linear relationships in data that simple linear models cannot capture. Without them, a multi-layer network would behave like a single-layer network.
    *   *Textual Evidence:* The red square with an 'S' curve symbol. Their strategic placement before output nodes (y1, y2) and potentially within hidden layer input calculation (h1) highlights their role.

*   **Error Calculation is Fundamental for Learning:** The network's performance is measured by the error between its predictions (y1, y2) and the true target values (t1, t2). This error is the basis for adjusting the network's parameters.
    *   *Textual Evidence:* "e1", "e2" (error signals); "t1", "t2" (target values); and the explicit blue arrows from y1 to t1 and y2 to t2 imply this comparison.

*   **Backpropagation as a Learning Mechanism:** The red feedback arrows from the output layer (y1, y2) to the hidden layer (h1) strongly suggest the process of backpropagation. This algorithm is used to effectively distribute the output error backward through the network, allowing for the calculation of gradients and subsequent updating of weights to reduce future errors. The specific feedback to h1 from both y1 and y2 shows how errors from multiple outputs can influence a single hidden neuron's learning.
    *   *Textual Evidence:* Red arrows from "y1" to "h1" and from "y2" to "h1"; "e1", "e2".

*   **Modular Design:** Neural networks are composed of interconnected processing units (neurons/nodes) that perform computations.
    *   *Textual Evidence:* The distinct circles representing x1, x2, h1, h2, h3, y1, y2.

**Document Context:**
This image would typically be found in academic, technical, or research documents discussing neural networks, deep learning, or machine learning algorithms. It serves as a foundational diagram to explain the basic architecture and operational principles of a multi-layer perceptron (a type of feedforward neural network). It would likely precede or accompany discussions on topics such as:
*   The building blocks of neural networks (neurons, layers, weights, biases).
*   The forward pass: how inputs are processed to generate outputs.
*   Activation functions and their role in introducing non-linearity.
*   The concept of supervised learning and the need for target values and error calculation.
*   The backpropagation algorithm: how errors are used to train the network by updating weights.
*   The mathematical representation of neuron activation and weight updates.

**Summary:**
This diagram illustrates a simplified model of a neural network, which is a powerful computational system inspired by the human brain, used for tasks like pattern recognition, prediction, and classification. Imagine it as a series of interconnected processing units, or "neurons," organized into layers.

At the very beginning, on the left, we have the **Input Layer**, represented by circles labeled **"x1"** and **"x2"**. These are where raw data or features (like characteristics of an image or numbers in a dataset) are fed into the network.

From the input layer, the information flows to the **Hidden Layer** in the middle, consisting of three neurons labeled **"h1"**, **"h2"**, and **"h3"**. Each input neuron (x1, x2) is connected to every hidden neuron (h1, h2, h3) by lines labeled with **'w' weights** (e.g., **"w₁₁"** from x1 to h1, **"w₂₃"** from x2 to h3). These weights are crucial; they determine the influence of one neuron's output on the next. A higher weight means a stronger influence.

After the hidden layer, the information moves to the **Output Layer** on the right, with two neurons labeled **"y1"** and **"y2"**. Similar to the previous step, each hidden neuron (h1, h2, h3) is connected to every output neuron (y1, y2) by lines labeled with **'u' weights** (e.g., **"u₁₁"** from h1 to y1, **"u₃₂"** from h3 to y2).

Before the final outputs `y1` and `y2` are produced, a special operation called an **activation function** is applied. These are shown as red square boxes containing an 'S' curve symbol. The diagram explicitly shows such activation functions (indicated by red 'S' boxes with red arrows) being applied on the paths leading to **"y1"** and **"y2"**. There's also one near **"h1"** from **"x1"**'s connection, which could signify an activation specific to that input to `h1` or a highlighted computational step. These functions introduce non-linearity, allowing the network to learn and model complex, real-world patterns.

Finally, the network produces its predictions, **"y1"** and **"y2"**. These predictions are then compared against the **Target Outputs**, labeled **"t1"** and **"t2"**, which are the true, desired values.

The difference between the network's prediction (y1, y2) and the target (t1, t2) is called the **error**, indicated by **"e1"** and **"e2"** (incoming red arrows to y1 and y2). These errors are incredibly important for "training" the network. The network uses these errors to learn by adjusting its weights.

The thick red arrows pointing backward from **"y1"** to **"h1"** and from **"y2"** to **"h1"** illustrate this learning process, specifically **backpropagation**. This means the error calculated at the output layer is propagated backward through the network. This backward flow of error information tells the network how much each weight (w's and u's) contributed to the error, and thus how to adjust them to make more accurate predictions in the future. The specific feedback from both `y1` and `y2` to `h1` shows that errors from both outputs contribute to the adjustment of the `h1` neuron's parameters.

In essence, this diagram visually explains how a neural network processes information in a forward direction (from inputs to outputs) and then learns by adjusting its internal weights based on errors propagating backward, making it a powerful tool for various intelligent tasks. The red arrow near `w₁₁` is a specific highlight on that weight, perhaps to emphasize its update or a particular calculation involving it.](images/adfd4c6cc847d1e5d60528724af78fa78af286c1cbffd1372d9c0e8d8812ffb9.jpg)

$\begin{array} { r } { \frac { \partial L } { \partial w _ { i j } } = \delta _ { j } x _ { i } \mathrm { , w h e r e } \delta _ { j } = \sum _ { k } \delta _ { k } w _ { j k } y _ { j } \big ( 1 } \end{array}$ − ????

If you want to examine the full derivation, please make use of the PDF we made available in the section: Backpropagation. A peek into the Mathematics of Optimization.