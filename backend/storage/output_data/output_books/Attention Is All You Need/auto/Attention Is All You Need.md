Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

# Attention Is All You Need

Ashish Vaswani∗ Google Brain avaswani@google.com

Noam Shazeer∗ Google Brain noam@google.com

Niki Parmar∗ Google Research nikip@google.com

Jakob Uszkoreit∗ Google Research usz@google.com

Llion Jones∗ Google Research llion@google.com

Aidan N. Gomez∗ † University of Toronto aidan@cs.toronto.edu

Łukasz Kaiser∗ Google Brain lukaszkaiser@google.com

Illia Polosukhin∗ ‡illia.polosukhin@gmail.com

# Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

# 1 Introduction

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h _ { t }$ , as a function of the previous hidden state $h _ { t - 1 }$ and the input for position $t$ . This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

# 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

# 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations $( x _ { 1 } , . . . , x _ { n } )$ to a sequence of continuous representations $\textbf { z } = ~ ( z _ { 1 } , . . . , z _ { n } )$ . Given $\mathbf { z }$ , the decoder then generates an output sequence $\left( y _ { 1 } , . . . , y _ { m } \right)$ of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

![## Image Analysis: af855240f9b9309607f19bbbc1d98f7b6fc795b504c1826e18f974c90887b7a2.jpg

**Conceptual Understanding:**
This image represents the complete conceptual architecture of the 'Transformer' model, a novel neural network architecture primarily designed for sequence transduction tasks, most famously in machine translation. The main purpose of this diagram is to visually delineate the intricate components and the data flow within the Transformer, offering a high-level yet detailed overview of its operational mechanism. It conveys key ideas such as the model's reliance on self-attention mechanisms, the incorporation of positional information into embeddings, and the distinct roles of its encoder and decoder components in processing input and generating output sequences respectively, all without relying on recurrent connections. This architecture allows for parallelization, making it computationally efficient for processing long sequences.

**Content Interpretation:**
The image illustrates the detailed architectural components and data flow within the Transformer model. It shows how an input sequence is processed by an Encoder and how an output sequence is generated by a Decoder, with crucial interactions between the two.

**Processes Shown:**
*   **Embedding and Positional Encoding:** Conversion of discrete tokens into continuous vector representations and the injection of sequential order information.
*   **Multi-Head Attention:** A mechanism allowing the model to attend to different parts of the input/output sequence in parallel, capturing long-range dependencies.
*   **Masked Multi-Head Attention:** A specific attention mechanism in the Decoder that ensures auto-regressive generation by preventing attention to future tokens.
*   **Feed Forward Networks:** Position-wise fully connected layers applied to the attention outputs for further transformations.
*   **Add & Norm:** The use of residual connections (Add) and layer normalization (Norm) to facilitate the training of very deep neural networks.
*   **Encoder-Decoder Attention:** The process by which the Decoder leverages the contextual representation from the Encoder to guide its output generation.
*   **Linear and Softmax Layers:** The final stages of converting the Decoder's hidden states into a probability distribution over the vocabulary for token prediction.

**Concepts/Systems Shown:**
*   **Encoder:** The component responsible for processing the input sequence and generating a rich contextual representation.
*   **Decoder:** The component responsible for generating the output sequence based on the Encoder's output and previously generated tokens.
*   **Transformer Model:** The entire system, characterized by its reliance on attention mechanisms and parallel processing without recurrence.

**Significance of Information:**
*   The 'Nx' labels for both Encoder and Decoder highlight the use of **stacked, identical layers**, enabling the model to learn complex, hierarchical representations.
*   The **'Positional Encoding'** is significant as it provides crucial sequential order information, a necessity given the parallel processing nature of attention. Without it, the model would lose track of word order.
*   **'Multi-Head Attention'** is central to the Transformer, allowing the model to capture diverse relationships and dependencies within and across sequences. Its different instantiations ('Masked' in the decoder, and cross-attention between Encoder and Decoder) are key to its functionality.
*   The **'Add & Norm'** blocks indicate the use of residual connections, which help prevent vanishing gradients in deep networks, and layer normalization, which stabilizes training by normalizing activations across features for each example.
*   The **'Linear' and 'Softmax'** layers signify the final output mechanism, converting the model's internal representations into interpretable probability distributions for token prediction.

**Key Insights:**
The image provides several key takeaways and insights into the Transformer model:

1.  **Attention-Centric Architecture:** The pervasive presence of 'Multi-Head Attention' and 'Masked Multi-Head Attention' as core building blocks in both the Encoder and Decoder demonstrates that attention mechanisms are fundamental to the Transformer, effectively replacing recurrence or convolutions for sequence modeling.
2.  **Parallel Processing with Explicit Positional Information:** The 'Positional Encoding' explicitly added to both 'Input Embedding' and 'Output Embedding' highlights the Transformer's ability to process sequence elements in parallel. This signifies a departure from sequential processing models (like RNNs) and the necessity of encoding position information separately.
3.  **Deep & Stable Architecture:** The 'Nx' notation for stacked layers, combined with the 'Add & Norm' layers, indicates a deep neural network architecture. The 'Add & Norm' layers are crucial for enabling the stable training of such deep models by mitigating issues like vanishing gradients through residual connections and normalizing activations.
4.  **Specialized Decoder for Auto-regressive Generation:** The 'Masked Multi-Head Attention' in the Decoder is a critical feature, enforcing an auto-regressive generation process. It ensures that the model can only attend to previously generated tokens, preventing it from 'seeing' future information during training and promoting accurate sequential prediction.
5.  **Effective Encoder-Decoder Communication:** The distinct 'Multi-Head Attention' layer in the Decoder that receives input from the Encoder (cross-attention) reveals how the Encoder's contextual understanding of the input sequence is directly utilized by the Decoder to guide the generation of the output sequence. This forms a powerful link between input comprehension and output synthesis.
6.  **Probabilistic Output for Sequence Prediction:** The final 'Linear' and 'Softmax' layers demonstrate that the Transformer's ultimate output is a probability distribution over the vocabulary, which is then used to select the next token in the sequence. This is a standard and interpretable way to handle sequence generation.

**Document Context:**
This image is presented as 'Figure 1: The Transformer - model architecture' within 'Section: 3 Model Architecture' of the document. Its placement immediately establishes it as the foundational visual representation of the core system being discussed. It provides a detailed, block-diagram view of the Transformer, enabling readers to grasp the conceptual flow and interaction of its various components before or in conjunction with textual explanations. This figure is crucial for understanding the subsequent technical descriptions and the underlying principles of how the Transformer processes information.

**Summary:**
The image displays a comprehensive diagram of the Transformer model architecture, which is a pivotal neural network structure in natural language processing, particularly for sequence-to-sequence tasks like machine translation. It is divided into two primary components: an Encoder on the left and a Decoder on the right. Both the Encoder and Decoder are shown as stacked, identical layers, indicated by the 'Nx' label, signifying that each block of operations is repeated 'N' times.

### Encoder (Left Side):
1.  **Inputs:** The process begins with raw 'Inputs' entering the model.
2.  **Input Embedding:** These inputs are first transformed into dense vector representations by the 'Input Embedding' layer.
3.  **Positional Encoding:** Simultaneously, 'Positional Encoding' is generated and added to the 'Input Embedding'. This step is crucial because it injects information about the order or position of tokens in the sequence, which is otherwise lost due to the Transformer's parallel processing nature.
4.  **Encoder Layer (Nx stack):** The combined embedding and positional encoding then proceed through 'N' identical Encoder layers. Each Encoder layer consists of two main sub-layers:
    *   **Multi-Head Attention:** This layer processes the input, allowing the model to jointly attend to information from different representation subspaces at different positions within the input sequence.
    *   **Add & Norm:** Following the 'Multi-Head Attention' layer, an 'Add & Norm' layer is applied. This layer performs a residual connection (adding the input of the Multi-Head Attention to its output) and then applies layer normalization, which helps in training deep networks.
    *   **Feed Forward:** The output then passes through a 'Feed Forward' network, which is a position-wise fully connected neural network that applies further transformations.
    *   **Add & Norm:** Another 'Add & Norm' layer follows the 'Feed Forward' network, similarly applying a residual connection and layer normalization.

### Decoder (Right Side):
1.  **Outputs (shifted right):** The Decoder's input begins with 'Outputs (shifted right)'. This refers to the previously generated output sequence, shifted by one position, ensuring that the prediction for a given position can only depend on known outputs at earlier positions.
2.  **Output Embedding:** These shifted outputs are converted into 'Output Embedding' vectors.
3.  **Positional Encoding:** Similar to the Encoder, 'Positional Encoding' is generated and added to the 'Output Embedding' to maintain sequence order information.
4.  **Decoder Layer (Nx stack):** The combined embedding and positional encoding then pass through 'N' identical Decoder layers. Each Decoder layer has three main sub-layers:
    *   **Masked Multi-Head Attention:** This is the first attention sub-layer, which is 'Masked' to prevent positions from attending to subsequent positions in the output sequence. This is essential for auto-regressive generation, ensuring the model predicts tokens based only on prior context. This is followed by an 'Add & Norm' layer.
    *   **Multi-Head Attention:** This second attention sub-layer performs Encoder-Decoder attention. It takes queries from the preceding 'Add & Norm' layer in the Decoder and attends to the keys and values derived from the *output of the Encoder stack*. This mechanism allows the Decoder to focus on relevant parts of the input sequence when generating each output token.
    *   **Add & Norm:** An 'Add & Norm' layer follows this second 'Multi-Head Attention' block.
    *   **Feed Forward:** Subsequently, a 'Feed Forward' network is applied.
    *   **Add & Norm:** A final 'Add & Norm' layer concludes the structure of each Decoder layer.

### Decoder Output Layers:
1.  **Linear:** The output from the final 'Add & Norm' layer in the last Decoder block passes through a 'Linear' layer, which projects the representation into a space typically the size of the vocabulary.
2.  **Softmax:** Finally, a 'Softmax' layer converts these scores into 'Output Probabilities' over the entire vocabulary, indicating the likelihood of each word being the next token in the sequence.](images/af855240f9b9309607f19bbbc1d98f7b6fc795b504c1826e18f974c90887b7a2.jpg)
Figure 1: The Transformer - model architecture.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

# 3.1 Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerN $\operatorname { y r m } ( x + \operatorname { S u b l a y e r } ( x ) )$ , where Sublayer $( x )$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d _ { \mathrm { m o d e l } } = 5 1 2$ .

Decoder: The decoder is also composed of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$ .

# 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum

![## Image Analysis: 83659013bee7db4de0b109708902294b8df739cc1a706ad1681aaeee9f6b35ae.jpg

**Conceptual Understanding:**
The image conceptually illustrates two closely related attention mechanisms used in neural networks, particularly within the Transformer architecture. The 'Scaled Dot-Product Attention' diagram represents the foundational process of computing attention weights by comparing query and key vectors, scaling them, and then applying these weights to value vectors. Its main purpose is to determine the relevance between elements in a sequence and produce a weighted sum of values. The 'Multi-Head Attention' diagram demonstrates an extension of this concept, where multiple 'Scaled Dot-Product Attention' mechanisms are run in parallel, allowing the model to attend to different aspects of the input simultaneously. Its main purpose is to enrich the model's representational capacity by jointly processing information from various subspaces, leading to a more comprehensive understanding of relationships within the data.

**Content Interpretation:**
The image displays two diagrams representing fundamental attention mechanisms in neural networks. The left diagram, 'Scaled Dot-Product Attention', illustrates the core computational flow for calculating attention, showing the sequence of operations from input queries, keys, and values to an attention-weighted output. The right diagram, 'Multi-Head Attention', demonstrates a more complex architecture where multiple 'Scaled Dot-Product Attention' mechanisms operate in parallel, their outputs concatenated and then linearly transformed, to enhance the model's ability to capture diverse relationships.

**Key Insights:**
1.  **Scaled Dot-Product Attention is the fundamental attention unit:** The diagram shows the sequential steps of MatMul (Q, K), Scale, Mask (opt.), SoftMax, and MatMul (with V), clearly outlining how queries, keys, and values interact to produce an attention-weighted output. This highlights the core computation for determining relevance and weighting values.
2.  **Multi-Head Attention processes information in parallel for richer representations:** The 'Multi-Head Attention' diagram, with its 'h' parallel 'Scaled Dot-Product Attention' blocks preceded by 'Linear' transformations for Q, K, V, demonstrates that running multiple attention mechanisms concurrently allows the model to learn different aspects of relationships within the data. This parallelism is key to its effectiveness.
3.  **Linear transformations are essential for projecting inputs into different subspaces:** The initial 'Linear' layers in Multi-Head Attention for V, K, and Q show that inputs are transformed before being fed into each attention head, enabling each head to operate on distinct representations.
4.  **Outputs from multiple attention heads are combined and re-projected:** The 'Concat' and final 'Linear' layers in Multi-Head Attention illustrate how the diverse information captured by individual attention heads is integrated into a unified representation, demonstrating the mechanism for combining various 

**Document Context:**
This image directly supports the document's section '3.2 Attention' by providing visual explanations of two key attention mechanisms: Scaled Dot-Product Attention and Multi-Head Attention. It details the internal workings and architectural components of these mechanisms, which are foundational to understanding how Transformer models process and relate information, thereby enhancing the reader's comprehension of the concepts discussed in the surrounding text.

**Summary:**
The image presents two distinct diagrams illustrating fundamental attention mechanisms in neural networks.

On the left, titled "**Scaled Dot-Product Attention**," a step-by-step process is shown, starting with three inputs: Query (Q), Key (K), and Value (V).
1.  The Query (Q) and Key (K) inputs first undergo a **MatMul** (Matrix Multiplication) operation.
2.  The result is then passed to a **Scale** operation, which adjusts the magnitude of the scores.
3.  Next, an optional **Mask (opt.)** step can be applied, used to selectively ignore certain values, for instance, in sequence processing.
4.  The output then proceeds to a **SoftMax** function, which transforms the scores into a probability distribution, creating the attention weights.
5.  Finally, a second **MatMul** (Matrix Multiplication) is performed. This operation combines the attention weights from the SoftMax layer with the original Value (V) input, producing a weighted sum of the values based on their relevance to the query and key. The output of this final MatMul is the result of the Scaled Dot-Product Attention.

On the right, titled "**Multi-Head Attention**," a more advanced and parallel architecture is depicted. This mechanism takes the same three general inputs: Value (V), Key (K), and Query (Q).
1.  Initially, each of the V, K, and Q inputs is independently processed by a **Linear** transformation layer. This happens for each of the `h` "attention heads" that run in parallel. The diagram shows three distinct "Linear" blocks for the V, K, and Q inputs that feed into the conceptual stack of attention mechanisms.
2.  The outputs from these initial Linear transformations for each head are then fed into a "**Scaled Dot-Product Attention**" block. The diagram shows multiple such blocks stacked, accompanied by the label "h", indicating that `h` separate and parallel Scaled Dot-Product Attention mechanisms are being computed. Each of these `h` "heads" can independently learn to focus on different parts of the input and capture different types of relationships.
3.  The individual outputs from all `h` parallel "Scaled Dot-Product Attention" blocks are then brought together through a **Concat** (Concatenation) operation, combining the information learned by each head into a single, larger representation.
4.  Finally, this concatenated output passes through one last **Linear** transformation layer, which projects the combined information back to the desired output dimension.
The overall Multi-Head Attention mechanism allows the model to process information from multiple representation subspaces simultaneously, significantly enhancing its ability to understand complex relationships within data compared to a single attention head.](images/83659013bee7db4de0b109708902294b8df739cc1a706ad1681aaeee9f6b35ae.jpg)
Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

# 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d _ { k }$ , and values of dimension √ $d _ { v }$ . We compute the dot products of the query with all keys, divide each by $\sqrt { d _ { k } }$ , and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$ . The keys and values are also packed together into matrices $K$ and $V$ . We compute the matrix of outputs as:

$$
\mathrm { A t t e n t i o n } ( Q , K , V ) = \mathrm { s o f t m a x } ( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } ) V
$$

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of √1 $\frac { 1 } { \sqrt { d _ { k } } }$ . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of $d _ { k }$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d _ { k }$ [3]. We suspect that for large values of $d _ { k }$ , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by $\frac { 1 } { \sqrt { d _ { k } } }$ .

# 3.2.2 Multi-Head Attention

Instead of performing a single attention function with $d _ { \mathrm { m o d e l } }$ -dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d _ { k }$ , $d _ { k }$ and $d _ { v }$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d _ { v }$ -dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$
\begin{array} { r } { \begin{array} { r l } & { \mathrm { M u l t i H e a d } ( Q , K , V ) = \mathrm { C o n c a t } ( \mathrm { h e a d } _ { 1 } , . . . , \mathrm { h e a d } _ { \mathrm { h } } ) W ^ { O } } \\ & { \qquad \mathrm { w h e r e ~ h e a d } _ { \mathrm { i } } = \mathrm { A t t e n t i o n } ( Q W _ { i } ^ { Q } , K W _ { i } ^ { K } , V W _ { i } ^ { V } ) } \end{array} } \end{array}
$$

Where the projections are parameter matrices $W _ { i } ^ { Q } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } }$ , W K ∈ Rdmodel×dk , W V ∈ Rdmodel×dv and $W ^ { O } \in \mathbb R ^ { h d _ { v } \times d _ { \mathrm { m o d e l } } }$ .

In this work we employ $h \ : = \ : 8$ parallel attention layers, or heads. For each of these we use $d _ { k } = d _ { v } = d _ { \mathrm { m o d e l } } / h \stackrel {  } { = } \dot { 6 } 4$ . Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

# 3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].   
The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.   
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $- \infty )$ all values in the input of the softmax which correspond to illegal connections. See Figure 2.

# 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

$$
\mathrm { F F N } ( x ) = \operatorname* { m a x } ( 0 , x W _ { 1 } + b _ { 1 } ) W _ { 2 } + b _ { 2 }
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d _ { \mathrm { m o d e l } } = 5 1 2$ , and the inner-layer has dimensionality $d _ { f f } = 2 0 4 8$ .

# 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d _ { \mathrm { m o d e l } }$ . We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax√ linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt { d _ { \mathrm { { m o d e l } } } }$

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.   

<table><tr><td>Layer Type</td><td>Complexity per Layer</td><td>Sequential Operations</td><td>Maximum Path Length</td></tr><tr><td>Self-Attention</td><td>O(n².d)</td><td>0(1)</td><td>0(1)</td></tr><tr><td>Recurrent</td><td>O(n·d²)</td><td>O(n)</td><td>0(n）</td></tr><tr><td>Convolutional</td><td>O(k ·n · d²)</td><td>0(1</td><td>O(logk(n))</td></tr><tr><td>Self-Attention (restricted)</td><td>O(r·n·d)</td><td>0(1)</td><td>0(n/r）</td></tr></table>

# 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d _ { \mathrm { m o d e l } }$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

In this work, we use sine and cosine functions of different frequencies:

$$
\begin{array} { r } { P E _ { ( p o s , 2 i ) } = s i n ( p o s / 1 0 0 0 0 ^ { 2 i / d _ { \mathrm { m o d e l } } } ) } \\ { P E _ { ( p o s , 2 i + 1 ) } = c o s ( p o s / 1 0 0 0 0 ^ { 2 i / d _ { \mathrm { m o d e l } } } ) } \end{array}
$$

where pos is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2 \pi$ to $1 0 0 0 0 \cdot 2 \pi$ . We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$ , $P E _ { p o s + k }$ can be represented as a linear function of P Epos.

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

# 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $( x _ { 1 } , . . . , x _ { n } )$ to another sequence of equal length $\left( z _ { 1 } , . . . , z _ { n } \right)$ , with $x _ { i } , z _ { i } \in \mathbf { \bar { \mathbb { R } } } ^ { d }$ , such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O ( n )$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$ , which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position. This would increase the maximum path length to $O ( n / r )$ . We plan to investigate this approach further in future work.

A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O ( n / k )$ convolutional layers in the case of contiguous kernels, or $O ( l o g _ { k } ( n ) )$ in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of $k$ . Separable convolutions [6], however, decrease the complexity considerably, to $\dot { O ( k \cdot n \cdot d + n \cdot d ^ { 2 } ) }$ . Even with $k = n$ , however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

# 5 Training

This section describes the training regime for our models.

# 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

# 5.2 Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

# 5.3 Optimizer

We used the Adam optimizer [20] with $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 8$ and $\epsilon = 1 0 ^ { - 9 }$ . We varied the learning rate over the course of training, according to the formula:

$$
l r a t e = d _ { \mathrm { m o d e l } } ^ { - 0 . 5 } \cdot \mathrm { m i n } ( s t e p _ { - } n u m ^ { - 0 . 5 } , s t e p _ { - } n u m \cdot w a r m u p _ { - } s t e p s ^ { - 1 . 5 } )
$$

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps $= 4 0 0 0$ .

# 5.4 Regularization

We employ three types of regularization during training:

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.   

<table><tr><td rowspan="2">Model</td><td colspan="2">BLEU</td><td colspan="2">Training Cost (FLOPs)</td></tr><tr><td>EN-DE</td><td>EN-FR</td><td>EN-DE</td><td>EN-FR</td></tr><tr><td>ByteNet [18]</td><td>23.75</td><td></td><td></td><td></td></tr><tr><td>Deep-Att + PosUnk [39]</td><td></td><td>39.2</td><td></td><td>1.0·1020</td></tr><tr><td>GNMT + RL [38]</td><td>24.6</td><td>39.92</td><td>2.3· 1019</td><td>1.4·1020</td></tr><tr><td>ConvS2S [9]</td><td>25.16</td><td>40.46</td><td>9.6·1018</td><td>1.5·1020</td></tr><tr><td>MoE[32]</td><td>26.03</td><td>40.56</td><td>2.0·1019</td><td>1.2 · 1020</td></tr><tr><td>Deep-Att + PosUnk Ensemble [39]</td><td></td><td>40.4</td><td></td><td>8.0·1020</td></tr><tr><td>GNMT +RL Ensemble [38]</td><td>26.30</td><td>41.16</td><td>1.8·1020</td><td>1.1 · 1021</td></tr><tr><td>ConvS2S Ensemble [9]</td><td>26.36</td><td>41.29</td><td>7.7 : 1019</td><td>1.2 · 1021</td></tr><tr><td>Transformer (base model)</td><td>27.3</td><td>38.1</td><td>3.3·1018</td><td></td></tr><tr><td>Transformer (big)</td><td>28.4</td><td>41.8</td><td>2.3 ·1019</td><td></td></tr></table>

Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P _ { d r o p } = 0 . 1$ .

Label Smoothing During training, we employed label smoothing of value $\epsilon _ { l s } = 0 . 1$ [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

# 6 Results

# 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than $1 / 4$ the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P _ { d r o p } = 0 . 1$ , instead of 0.3.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty $\alpha = 0 . 6$ [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length $+ 5 0$ , but terminate early when possible [38].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU 5.

# 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.   

<table><tr><td rowspan="2"></td><td rowspan="2">N dmodel</td><td rowspan="2">dff</td><td rowspan="2">h</td><td rowspan="2">dk</td><td rowspan="2">d</td><td rowspan="2">Pdrop</td><td rowspan="2">es</td><td rowspan="2">trains</td><td rowspan="2">PPL</td><td rowspan="2">BLeU</td><td rowspan="2">px10s</td></tr><tr><td></td></tr><tr><td>base</td><td>6</td><td>512</td><td>2048</td><td>8</td><td>64</td><td>64</td><td>0.1 0.1</td><td>100K</td><td>4.92</td><td></td><td>25.8</td><td>65</td></tr><tr><td rowspan="3">(A)</td><td></td><td></td><td>1</td><td>512</td><td>512</td><td></td><td></td><td></td><td>5.29</td><td></td><td>24.9</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>128</td><td>128</td><td></td><td></td><td></td><td>5.0</td><td>25.5</td><td></td></tr><tr><td></td><td></td><td>46 32</td><td>16</td><td>16</td><td></td><td></td><td></td><td>5.01</td><td></td><td></td><td></td></tr><tr><td>(B)</td><td></td><td></td><td></td><td></td><td>1632</td><td></td><td></td><td></td><td></td><td>5.16</td><td>25.4 25.4</td><td>5860</td></tr><tr><td rowspan="4">(C)</td><td>248</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.11</td><td></td><td>23.7</td><td>36 50</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.19 4.88</td><td>25.3 25.5</td><td>80</td></tr><tr><td>256</td><td></td><td></td><td>32</td><td>32</td><td></td><td></td><td></td><td>5.75</td><td>24.5</td><td>28</td></tr><tr><td>1024</td><td></td><td></td><td>128</td><td>128</td><td></td><td></td><td></td><td>4.66</td><td>26.0</td><td>168</td></tr><tr><td></td><td>1024</td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.12</td><td>25.4</td><td>53</td></tr><tr><td rowspan="3">(D)</td><td></td><td>4096</td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.75</td><td></td><td>26.2</td><td>90</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td>5.77</td><td>24.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.2</td><td>0.0</td><td></td><td>4.95</td><td>25.5</td><td></td><td></td></tr><tr><td rowspan="2">(E)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2</td><td></td><td>5.47</td><td>25.7</td><td></td></tr><tr><td colspan="2">positional embedding instead of sinusoids</td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.92</td><td></td><td></td><td></td></tr><tr><td>big</td><td>6</td><td>1024</td><td>4096</td><td>16</td><td></td><td></td><td>0.3</td><td></td><td>300K</td><td>4.33</td><td>25.7 26.4</td><td>213</td></tr></table>

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

In Table 3 rows (B), we observe that reducing the attention key size $d _ { k }$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

# 6.3 English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

We trained a 4-layer transformer with $d _ { m o d e l } = 1 0 2 4$ on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length $+ 3 0 0$ . We used a beam size of 21 and $\alpha = 0 . 3$ for both WSJ only and the semi-supervised setting.

Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)   

<table><tr><td>Parser</td><td>Training</td><td>WSJ 23 F1</td></tr><tr><td>Vinyals &amp; Kaiser el al. (2014) [37] Petrov et al. (2006) [29]</td><td>WSJ only, discriminative WSJ only, discriminative</td><td>88.3 90.4</td></tr><tr><td>Zhu et al. (2013) [40]</td><td>WSJ only, discriminative</td><td>90.4</td></tr><tr><td>Dyer et al. (2016) [8]</td><td>WSJ only, discriminative</td><td>91.7</td></tr><tr><td>Transformer (4 layers)</td><td>WSJonly, discriminative</td><td>91.3</td></tr><tr><td>Zhu et al. (2013) [40]</td><td>semi-supervised</td><td>91.3</td></tr><tr><td>Huang &amp; Harper (2009) [14]</td><td>semi-supervised</td><td>91.3</td></tr><tr><td>McClosky et al. (2006) [26]</td><td>semi-supervised</td><td>92.1</td></tr><tr><td>Vinyals &amp; Kaiser el al. (2014) [37]</td><td>semi-supervised</td><td>92.1</td></tr><tr><td>Transformer (4 layers)</td><td>semi-supervised</td><td>92.7</td></tr><tr><td>Luong et al. (2015) [23]</td><td>multi-task</td><td>93.0</td></tr><tr><td>Dyer et al. (2016) [8]</td><td>generative</td><td>93.3</td></tr></table>

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of 40K sentences.

# 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

# References

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.   
[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.   
[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016. [5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014. [6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016. [7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014. [8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.   
[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.   
[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.   
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.   
[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.   
[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.   
[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832–841. ACL, August 2009.   
[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.   
[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.   
[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.   
[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.   
[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.   
[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.   
[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.   
[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.   
[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.   
[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.   
[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.   
[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152–159. ACL, June 2006.   
[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.   
[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.   
[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July 2006.   
[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.   
[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.   
[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.   
[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.   
[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates, Inc., 2015.   
[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.   
[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.   
[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.   
[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.   
[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.   
[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434–443. ACL, August 2013.

# Attention VisualizationsInput-Input Laye

![## Image Analysis: 64cc9b09229cb709b675ce7867ba9e99d0a172887d26a07d2a95dfbc02a04443.jpg

**Conceptual Understanding:**
This image conceptually illustrates the attention mechanism of a neural network, specifically how it processes and understands contextual relationships within a sentence. Its main purpose is to visualize the model's ability to capture long-distance dependencies, demonstrating how the word "making" semantically connects to "more difficult" despite other words appearing between them. The key idea conveyed is the power of self-attention in discerning meaningful linguistic links beyond immediate word proximity.

**Content Interpretation:**
The image displays the attention weights from the word "making" to other words in the sentence "It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult." The processes shown involve the self-attention mechanism within a neural network encoder, specifically how it links words. The key concepts are long-distance dependency identification and the role of multiple attention heads. The strong, distinct colored lines connecting "making" to "more" and "difficult" signify that multiple attention heads are collaboratively identifying this semantic relationship, which is crucial for contextual understanding.

**Key Insights:**
The main takeaway is that attention mechanisms, particularly multi-head attention, effectively identify and utilize long-distance dependencies in natural language. This is evidenced by the distinct connections from "making" to "more" and "difficult" in the transcribed sentence, showing the model's ability to semantically link words that are far apart. The visualization also suggests that different attention heads might focus on various aspects of a dependency, contributing to a robust understanding of the phrase "making...more difficult."

**Document Context:**
This image serves as a direct illustration for the document's section on "Attention Visualizations" and "Input-Input Layer." It provides a concrete example of how the attention mechanism, a core component of Transformer models, functions to resolve long-distance dependencies in text. The image's demonstration of "making" attending to "more difficult" aligns with the accompanying text that describes it as an example of an attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6, completing the phrase 'making...more difficult'.

**Summary:**
This image visualizes the attention mechanism of a neural network, specifically demonstrating how the word "making" in a sentence attends to other words. The image shows two identical vertical text sequences: "It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS> <pad> <pad> <pad> <pad> <pad> <pad>" The word "making" in the top sequence is highlighted by a grey vertical bar, indicating it's the focal point of the attention. From this highlighted "making," multiple colored lines, representing different attention heads, extend to various words in the bottom sequence. The strongest and most numerous lines connect "making" to "more" and "difficult," using colors such as purple, light blue, light green, orange, light brown, and red. This visual mapping illustrates that when the model processes "making," it strongly attends to "more" and "difficult" to understand the complete phrase "making...more difficult," even across several intervening words. This demonstrates the model's capability to capture long-distance linguistic dependencies.](images/64cc9b09229cb709b675ce7867ba9e99d0a172887d26a07d2a95dfbc02a04443.jpg)
Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.

![## Image Analysis: a4189dcdb921b9c52ed7e5343a9814769640f53d0218c038a054f93258bff351.jpg

**Conceptual Understanding:**
This image conceptually represents an attention mechanism within a neural network, likely a transformer model, illustrating how different parts of an input sequence (words in a sentence) are weighted or "attended to" when processing another part of the sequence. The main purpose of this visualization is to demonstrate and make interpretable the model's ability to understand complex linguistic relationships, specifically focusing on anaphora resolution. It conveys the key idea that attention mechanisms can precisely identify the antecedents of pronouns and the nouns they modify, thereby contributing to a deeper understanding of sentence semantics.

**Content Interpretation:**
The image illustrates the attention mechanism within a neural network, specifically highlighting how different attention heads process linguistic relationships. The core process shown is the calculation and visualization of attention weights between words in a sentence across two layers.

In the **top visualization**, the processes are:
*   **Distributed Attention:** Each word in the lower sentence layer is shown to attend to multiple words in the upper sentence layer, represented by faint purple lines. This indicates a broad contextual understanding where many words contribute to the meaning of others.

In the **bottom visualization**, the processes are highly focused on the word 'its':
*   **Anaphora Resolution:** A strong purple line connects 'its' (lower layer) to 'Law' (upper layer), demonstrating the model's ability to identify the antecedent of a pronoun. The word 'Law' in the lower layer is also highlighted in purple, reinforcing its role as the referent.
*   **Possessive Modifier Identification:** A strong brown line connects 'its' (lower layer) to 'application' (lower layer), indicating the model's understanding that 'its' modifies 'application'. The word 'application' in the lower layer is highlighted in brown, emphasizing this direct grammatical relationship.

The significance of the information presented is the demonstration of specialized attention heads. The contrast between the diffuse attention in the top panel and the sharp, focused attention on 'its' in the bottom panel reveals that specific attention heads can be highly effective at resolving complex linguistic tasks like anaphora and identifying grammatical dependencies.

**Key Insights:**
The image provides several key takeaways regarding neural network attention mechanisms:
1.  **Anaphora Resolution Capability:** Attention heads can effectively resolve anaphora by establishing strong connections between a pronoun (e.g., 'its') and its referent (e.g., 'Law'). This is evidenced by the prominent purple line from 'its' to 'Law' in the bottom panel.
2.  **Identification of Local Grammatical Dependencies:** Beyond long-range anaphora, attention also captures immediate grammatical relationships, such as a possessive pronoun modifying a noun (e.g., 'its application'). This is shown by the strong brown line from 'its' to 'application' within the same layer.
3.  **Specialization of Attention Heads:** Different attention heads within a transformer model can specialize in distinct linguistic tasks. The diffuse attention in the top panel (Head 5 general attention) contrasts sharply with the highly focused attention from 'its' in the bottom panel (Heads 5 and 6 specific attention), illustrating this specialization.
4.  **Interpretability of Model Decisions:** Visualizing attention weights allows researchers to understand *how* the model is making connections and deriving meaning from text, providing insights into its internal workings and supporting debugging or model improvement efforts. The 'sharpness' of the attentions for 'its' indicates a confident and specific linguistic decision.

**Document Context:**
This image is directly relevant to the document's section on "Attention Visualizations Input-Input Layer." It serves as a crucial visual example demonstrating how attention mechanisms, particularly within specific heads of a transformer layer, function and can be interpreted. It specifically illustrates the process of anaphora resolution, where a pronoun ('its') is linked to its antecedent ('The Law') and the noun it modifies ('application'). This visualization helps to demystify a complex internal process of neural networks, making the model's linguistic processing more understandable for readers within the broader narrative of model interpretability and analysis.

**Summary:**
This image presents an attention visualization from a neural network, specifically showing how different words in a sentence attend to each other. The visualization is divided into two main panels, both displaying the same sentence across two layers: "The Law will never be perfect, but its application should be just, this is what we are missing in my opinion <EOS> <pad>". The sentence is shown once in an upper layer and once in a lower layer, with lines connecting words between these layers or within the same layer, representing attention weights.

The **top panel** illustrates the "Full attentions for head 5" of the model. Here, each word in the lower sentence layer has faint purple lines extending to many words in the upper sentence layer. This diffuse pattern indicates that this particular attention head might be considering a broad range of contextual words when processing each input word, showing a general distribution of attention across the sentence.

The **bottom panel** provides "Isolated attentions from just the word ‘its’ for attention heads 5 and 6," focusing on the pronoun "its" from the lower sentence layer. In this panel, the attention patterns are significantly sharper and more focused, primarily originating from the word "its" in the lower sentence layer (which has a faint gray background). There are two very prominent connections:
1.  A strong purple line connects "its" from the lower sentence layer to "Law" in the upper sentence layer. Notably, "Law" in the lower sentence layer is also highlighted with a purple background, visually reinforcing this connection. This link suggests that the model understands "The Law" as the antecedent or referent for the pronoun "its."
2.  A strong brown line connects "its" within the lower sentence layer to "application" also within the lower sentence layer. The word "application" in the lower sentence layer is highlighted with a brown background. This connection indicates that "its" is directly modifying the word "application" in the phrase "its application."

This detailed visualization demonstrates the model's ability to perform anaphora resolution by linking "its" to "The Law" and also to identify the direct grammatical dependency between "its" and "application." The sharpness of these specific attention links, especially in contrast to the more diffuse pattern in the top panel, highlights the specialized linguistic function of these attention heads and their role in accurately interpreting sentence meaning. It serves as clear evidence for how internal attention mechanisms contribute to a model's comprehension of complex linguistic phenomena like anaphora.](images/a4189dcdb921b9c52ed7e5343a9814769640f53d0218c038a054f93258bff351.jpg)
Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.

![## Image Analysis: 97a537463e638e60d0083c5b6971e794ff2092a3ba511646414d12148e7d14c9.jpg

**Conceptual Understanding:**
This image conceptually represents attention weight distributions within a transformer model's self-attention mechanism, specifically showing two different "attention heads" from an encoder layer. The main purpose is to visualize which words in a sentence a given word "attends" to, revealing the learned relationships and dependencies captured by different attention heads. It demonstrates how different heads specialize in different linguistic tasks. Key ideas communicated are multi-head attention, the capture of syntactic and semantic relationships, and the interpretability of neural network models through visualization.

**Content Interpretation:**
The image displays two distinct attention mechanisms (or "heads") operating on the same input sentence. Each visualization shows how words in the bottom row (query words) "attend" to words in the top row (key/value words). The thickness and color intensity of the lines indicate the strength of the attention weight.

Attention Head 1 (Green Visualization - Top):
- The word "application" in the bottom row strongly attends to "its" and "application" itself in the top row. It also has strong connections to "Law" and "perfect".
- The word "missing" in the bottom row strongly attends to a wide range of words in the top row including "this", "is", "what", "we", "are", "missing", "in", "my", "opinion". This suggests it gathers context from a broad segment of the latter part of the sentence.
- Other notable connections: "Law" attends to "Law". "perfect" attends to "perfect". "just" attends to "application", "should", "be", "just", "this". "opinion" attends to "opinion".

Attention Head 2 (Red Visualization - Bottom):
- Many words in the bottom row primarily attend to themselves in the top row (e.g., "The" to "The", "Law" to "Law", "will" to "will", "never" to "never", "perfect" to "perfect", "but" to "but", "its" to "its", "should" to "should", "be" to "be", "just" to "just", "this" to "this", "is" to "is", "what" to "what", "we" to "we", "are" to "are", "missing" to "missing", "in" to "in", "my" to "my", "opinion" to "opinion").
- There are also strong connections to adjacent words. For instance, "Law" attends to "The". "be" attends to "never". "perfect" attends to "be". "application" attends to "its". "should" attends to "application". "be" (second instance) attends to "should". "what" attends to "is". "we" attends to "what". "are" attends to "we". "missing" attends to "are". "my" attends to "in". "opinion" attends to "my".
- Punctuation and Special Tokens: The comma "," attends to "perfect". The dash "-" attends to "just". "<EOS>" attends to "opinion". "<pad>" attends to "<EOS>".

Textual Evidence:
- Green Head: The thick lines connecting bottom "application" to top "its" and "Law", and bottom "missing" to top "this", "is", "what", "we", "are", "missing", "in", "my", "opinion" clearly demonstrate these distinct attention patterns. The word "application" is a key noun in the sentence, and the head focuses on its antecedents and modifiers. "missing" is a key concept, and its broad attention shows an attempt to grasp its full context.
- Red Head: The thick lines generally running almost vertically from a word in the bottom row to the same word in the top row (e.g., "perfect" to "perfect", "should" to "should") or to its immediate left neighbor (e.g., "Law" to "The", "application" to "its") provide direct evidence for local/identity attention.

**Key Insights:**
Main Takeaways/Lessons:
- Attention Head Specialization: Different attention heads within a transformer model learn to perform distinct tasks or capture different types of linguistic information. The green head appears to capture more semantic or long-range dependencies, while the red head focuses on local or identity relationships.
- Interpretability of Neural Networks: Visualizing attention weights provides a powerful method for understanding what parts of the input a model focuses on when processing specific tokens, making the model less of a "black box."
- Sentence Structure Understanding: The observed patterns show that the model learns aspects related to the syntactic and semantic structure of the sentence.

Conclusions/Insights:
The two heads presented demonstrate that transformer models do not create a single, undifferentiated representation. Instead, they use multiple "lenses" (heads) to parse the input, each focusing on different aspects of the sentence structure and meaning. This multi-faceted approach contributes to their powerful language understanding capabilities.

Textual Evidence:
The verbatim words "The Law will never be perfect, but its application should be just - this is what we are missing in my opinion <EOS> <pad>" serve as the core data. The specific connections observed between these words (e.g., bottom "application" connecting to top "Law" in green, or bottom "perfect" connecting to top "perfect" in red) are the direct evidence for the conclusions about head specialization and the types of relationships learned.

**Document Context:**
This image is crucial for the "Attention Visualizations" section as it directly illustrates how attention mechanisms work in practice. It provides concrete examples of how different attention heads, particularly from encoder self-attention at layer 5 of 6, learn to focus on different parts of a sentence to derive meaning. The accompanying document context text explicitly states this purpose: "Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks." The image visually confirms this, supporting the argument that attention is interpretable and task-specific within transformer models.

**Summary:**
This image displays two "attention visualization" diagrams, each representing the focus of a different "attention head" within a transformer model, specifically from the encoder self-attention at layer 5 of 6. Both diagrams analyze the same input sentence: "The Law will never be perfect, but its application should be just - this is what we are missing in my opinion <EOS> <pad>". Each visualization consists of two identical rows of words, representing the input layer (top) and the layer where attention is being queried (bottom). Lines connect words from the bottom row to words in the top row; the thickness and color intensity of these lines indicate the strength of the attention weight, showing which words an "output" word (from the bottom row) is "attending" to in the "input" (top row).

The top visualization (green lines) shows an attention head that appears to capture more semantic or long-range dependencies. For instance, the word "application" in the bottom row strongly attends to "its" and "Law" in the top row, indicating it's linking to the subject or modifiers of "application". Significantly, the word "missing" in the bottom row strongly attends to a broad segment of the sentence in the top row, including "this", "is", "what", "we", "are", "missing", "in", "my", and "opinion". This suggests this head is gathering extensive contextual information for key conceptual words.

In contrast, the bottom visualization (red lines) depicts an attention head that largely focuses on local relationships or identity mapping. Many words in the bottom row, such as "perfect", "should", "just", and "missing", predominantly attend to themselves in the top row with very strong, almost vertical lines. Additionally, this head captures strong connections to immediately preceding words, like "Law" attending to "The", "application" attending to "its", and "missing" attending to "are". This suggests this head is primarily concerned with local syntactic structure, word identity, and immediate neighborhood context within the sentence.

Together, these two visualizations clearly demonstrate the concept of multi-head attention, where different attention heads learn to perform distinct, specialized tasks. One head (green) focuses on broader, potentially semantic relationships, while the other (red) concentrates on local, short-range, or identity-based connections. This illustrates how complex models derive meaning by simultaneously analyzing different facets of sentence structure.](images/97a537463e638e60d0083c5b6971e794ff2092a3ba511646414d12148e7d14c9.jpg)
Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.