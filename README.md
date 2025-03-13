*For instructions on how to run my custom language model and tokenizer, scroll to the bottom*

# LLM Research: Final Report

### Introduction
For the duration of my Independent Study course, I have been tasked with researching the architecture of Large Language Models, a cutting-edge technology in the Artificial Intelligence landscape. LLM study has been prevalent for many years; however, it is only until the development of transformer architecture that innovations in this field have skyrocketed.

### Neural Networks
Before diving deep into Large Language Models, I had to educate myself on neural networks and create a strong foundation to build my knowledge off of. Artificial neural networks are digital data structures used to digitally mimic neural interactions within the human brain. They are represented as a series of connected nodes which communicate information to each other based on signals. Each digital neuron or node in the network communicates with others based on unique traits of the nodes, called a weight (and sometimes bias).

A neural network can easily be visualized as a directed acyclic graph (DAG) where vertices represent nodes and edges represent the one-way communication the neurons communicate through to one another. However, neural networks in practice are represented as matrices (3Blue1Brown, 2017). The group of neurons that take in input to process from the outside world is called an input layer. Once the input layer receives information from outside the network, this information is propagated to the rest of the “layers” in the network. This process is called a forward pass. 

However, there is a major issue at this point with how the neural network processes data. At this point, all of the neurons’ weights and biases are completely arbitrary. We need a process of slightly adjusting these weights and biases so that over time, the network gets better at performing any arbitrary task such as pattern recognition. The process of slightly adjusting the parameters of the network and slightly improving the output is called gradient descent. The difference between the actual output from the intended output is called the loss of the output. By slightly adjusting the neurons’ parameters and getting the loss, you can partially differentiate at any change in the network to estimate the local minimum for that specific change to the overall loss score. One step of locally minimizing the loss at every neuron is called the backward pass (3Blue1Brown, 2017).

The forward pass and the backward pass together make up an entire multi-layer perceptron (MLP). When testing or validating a network’s accuracy, only the forward pass is used and its outputs go unanalyzed. Training and validation steps are almost always performed with matrix operations as they are easily parallelizable and therefore they can be run on a GPU instead of a CPU. Running calculations on a GPU makes computations like these extremely efficient.

### Neural Networks: Kolmogorov Arnold Networks
Because of this, each activation function of a neuron is a simple linear function of y = mx + b where m is the weight of the neuron, and b is the bias of the neuron, while x is the input value and y is the output. In a very simplistic view, we are combining a large quantity of linear functions to predict the output of a non-linear unknown function. However, with Kolmogorov Arnold Networks, we may need much fewer neurons to achieve similar results. In a paper written in 2024 by the Massachusetts Institute of Technology, California Institute of Technology, and Northeastern University, the researchers propose an alternative to the Multi-Layer Perceptron (MLP) called Kolmogorov-Arnold Networks (KAN). KANs’ main difference is they represent a neuron’s activation function as spline, which is defined as a series of n-degree polynomials. As linear activation functions are only straight lines, KANs can be any number of continuous functions given specific parameters regarding the polynomial coefficients of the splines (Liu et al., 2020).

However, KANs have one major flaw. Compared to MLPs, they are significantly slower in reducing the loss over time, although they use significantly fewer neurons overall. I am still waiting for an update as this issue has not been improved (Liu et al., 2020).

### Large Language Models (LLMs)
Now that I have completed my review of neural networks, I can begin to share my findings on LLMs. LLMs are auto-regressive natural language processors. Auto-regressive describes AI models that try to predict future events based on past events. Natural language processors (NLP) are models designed to manipulate, interpret, and generate human language. Before ChatGPT, LLM research has been studied for well over a decade, however many of GPT’s predecessors all suffered from similar shortcomings: context length. Context length describes a language model’s ability to call on previous information when dealing with new information. The farther back in its “memory” it can base its decisions on the longer its context length. Many researchers in the field also refer to this memory itself as the “context window”. One of the most popular language model architectures in the early 1980s was called Recurrent Neural Networks (Sherstinsky, 2018).

### Transformer Architecture: Introduction
The large majority of LLMs today are built around the transformer architecture, mostly unchanged since its conception in 2017. A paper titled “Attention Is All You Need” introduced architecture as a way to translate text from English to various other languages and vice-versa. It wasn’t until a paper released by Open AI in 2020 titled “Language Models are Few-Shot Learners” that the public started to realize the emergent properties of NLPs from scaling them with the transformer architecture. In the creation of GPT-3, Open AI researchers scraped a large portion of the open web to train the model on natural language and discovered that these models are capable of not only mimicking human language but human reasoning capabilities too (Open AI, 2020).

### Transformer Architecture: Tokenization
The transformer architecture starts with the tokenizer, a preprocessing technology used by machines to make text more understandable. Tokenization involves encoding sentences as a sequence of decimal values. These decimal values can correspond to individual letters but often correspond to longer sequences of characters, like sections of words or entire words. Normally tokens don’t represent more than a whole word itself. Transformers do not process sentences normally as humans do but rather they process tokens. This allows for LLMs to better understand individual words and their meaning in different contexts.

The most popular tokenizer TikToken, developed and open-sourced by Open AI, is training using Byte-Pair Encoding (BPE). BPE was initially invented in the 1990s as a form of data compression. How it works by providing the tokenizer with a large amount of training text, the tokenizer initially assigns each character to a value (usually a UTF encoding) and then iteratively replaces the most common pairings of numbers with a new number appended to the tokenizer lookup table of tokens to pairings of tokens. It’s an upside down tree where pairs of symbols are then merged into larger values and it is continuously merging these most common pairings. The number of iterations in training the tokenizer is arbitrary, however eventually, when the frequency of pairs is one-to-one with the frequency of the characters they make up, then the compression is no longer efficient (Karpathy, 2024).

Using the videos on transformers by Andrej Karpathy, I constructed my tokenizer in Python. The videos can be found [here](https://www.youtube.com/watch?v=kCc8FmEb1nY). My custom tokenizer code can be found [here](https://github.com/gg-blake/cs478/blob/main/gpt/tokenizer.py). Once I completed the tokenizer, I immediately began training it on a couple of data sources such as Open AI’s Open Web Text, which consists of vast amounts of scraped Reddit forum data (Karpathy, 2024). I also made a custom scraper to scrape UMass Boston’s website for text. Both data sources were massive in size and took multiple days to train on however my results for both were unsatisfactory. The main issue with training a tokenizer is that the BPE algorithm is inherently single-threaded and can’t be batched in any way. So no matter what machine I run it on, it will be bottlenecked by a single core’s performance. After this realization, I decided to proceed with the rest of my LLM studies with TikToken. Tiktoken has a variety of pre-trained tokenizer models and its BPE training implementation is also written in Rust as opposed to Python which is significantly slower. You can find the repo for TikToken [here](https://github.com/openai/tiktoken).

However, during my struggles to improve the performance of my tokenizer, I was able to make improvements to my own tokenizer. Most notable, was the way that I calculated the frequencies of each token pair. In my implementation, I was able to update the frequencies of the pairs in place while merging so what was once a linear time computation task became a constant-time computation.

### Transformer Architecture: Self-Attention
The first step in transformer architecture is embedding the tokens. A transformer language model contains a lookup table that maps tokens to an n-dimensional vector. These lookup tables are called embeddings and a model can have multiple different embeddings that give encodes different meanings into tokens.

The transformer architecture starts with mapping all the tokens to a sequence of positional embeddings and token embeddings. Once the input tokens have been mapped to a sequence of position and token embeddings, each token and position embedding are added element-wise and fed into the transformer block.

The most important part of the transformer architecture that was introduced in “Attention Is All You Need” is self-attention. Is a process in which the individual embeddings in the sequence affect the embeddings of the surrounding embeddings. Just like how words in the human language have different meanings based on the context in which they are used, embeddings in the presence of other embeddings will affect their meaning and the meaning of the embeddings around them in the input sequence. In a head of attention, an individual input embedding is given a key, query, and value vector, of the same dimensions. In the simplest terms, the key vector is affected by the surrounding embeddings in the input embedding sequence. The query vector is the vector that affects other embeddings’ key vectors in the embedding sequence. The value is the randomized intermediary tensor that amalgamates or ties together the key and query vectors. Applying these three tensors to the input embedding yields a self-attended vector that has a new position in this n-dimensional vector space that is affected by the presence of other embeddings in the input sequence. This calculation of attention is “self-attending” because it does not require the resolving of values other than its own key, query, and value parameters, and so it can be easily parallelized in what is called Multi-Headed Attention (Vaswani et al., 2017, 4).

$${Attention}(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$
$${softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

For diffusion (image) models, a special type of attention can be applied called cross-attention. Cross-attention uses the key and query vectors for the embeddings from a separate source. This is the original design proposed in the original paper for language translation where the key and query vectors are taken from a different language like French or Spanish and fed into a Multi-Headed Attention block where the value comes from the target language (i.e. English).

Additionally, multiple Multi-Headed Attention blocks can be chained one after the other to improve the results of the output as well. However, longer chains of Multi-Headed Attention blocks make it harder to train from the backward pass, so the output of each Multi-Headed Attention block is added to the input embeddings in what are called residual connections. In a paper called “Deep Residual Learning for Image Recognition”, they explain how this significantly improves training performance (He et al., n.d., 2)

![image.png](https://github.com/gg-blake/cs478/blob/main/image.png?raw=True)

Furthermore, once the residual connections are built, we must perform what is called layer normalization to the layers to “clean” the data or rid the data of noise so to speak. In the paper “Layer Normalization” the researchers propose the following solution called layer normalization.

$$y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta$$

Where and are learnable parameters like weights and biases of the neural network (Lei Ba et al., 2016, 2). If you look closely, the formula resembles the same shape which is a linear equation.

I implemented a GPT-like language model using the transformer architecture in Python. Andrej Karpathy was a great resource for building this model, big thanks to him for helping me wrap my head around these really complex machine-learning concepts. My code for the model can be found [here](https://github.com/gg-blake/cs478/blob/main/gpt/model.py). The Pytorch checkpoints can be found [here](https://github.com/gg-blake/cs478/tree/main/gpt/lm_models/openwebtext).

### Transformer Architecture: Bidirectional Encoder Representations (BERT)**
Following the construction of a GPT-like language model, it is not ready to be a helpful assistant. It requires fine tuning, and training on specific types of data so that the model predicts what a useful assistant would say. There are many different ways to do this, however, for my research, I focused on BERT. Bidirectional Encoder Representations (BERT) is a framework for training language models to understand human languages, specifically in the context of a question-and-answer or fill-in-the-blanks. There are four main types of BERT training methods, the one I am using is the first type, in which text is provided to the model, and randomly a subsequence of tokens is replaced with a special “mask” token to represent a blank space that should have text. The model, instead of constantly generating new text, is now required to choose a finite set of tokens to fill in the masked tokens (Devlin et al., 2018, 2). The loss is calculated using a common equation called Cross Entropy. At the time of writing this report, I am in the process of building the training environment for my model and generating quality testing data from Open AI’s Open Web Text. You can find my current code for this [here](https://github.com/gg-blake/cs478/blob/main/gpt/train_bert.py).

### Conclusion
During my independent study, I learned about the underlying architecture of modern LLMs and Transformer-based language models like ChatGPT. I learned about how tokenizers break text into numerical chunks for transformers. Later on, I saw how transformers embed these tokens into higher-dimensional vector spaces. I learned how these high-dimensional embedding subspaces each contain their own unique human concepts like gender and location. Additionally, I learned about Self-Attention, and how it allows the context of many tokens in a sequence to be evaluated in parallel to each other. I also touched on the key differences between transformer encoders and decoders and applications for image generation. Lastly, I discussed a popular training framework for transformers called BERT and my current efforts to implement this into my custom language model.

So far, I have gained an immense amount of knowledge in the cutting-edge field in such a short amount of time. I have become confident enough to read research papers as they are released regarding new breakthroughs on the LLM front. As I learn more, I believe I have reached the bottom of the Dunning Kruger Pit in a sense; there is still so much to learn and I am so grateful and proud of myself for taking the leap and educating myself on a subject that is still in its infancy. I hope to someday be the people in these papers, contributing to what I believe is a new era of computing.

### References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _arxiv.org_. <https://doi.org/10.48550/arXiv.1810.04805>

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. _arxiv.org_. <https://doi.org/10.48550/arXiv.1512.03385>

Karpathy, A. (2023, January 17). _Let's build GPT: from scratch, in code, spelled out._ YouTube. Retrieved December 26, 2024, from <https://www.youtube.com/watch?v=kCc8FmEb1nY>

Karpathy, A. (2024, February 20). _Let's build the GPT Tokenizer_. YouTube. Retrieved December 26, 2024, from <https://www.youtube.com/watch?v=zduSFxRajkE>

Lei Ba, J., Ryan Kiros, J., & Hinton, G. E. (2016). Layer Normalization. _arxiv.org_. <https://doi.org/10.48550/arXiv.1607.06450>

Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2020). KAN: Kolmogorov-Arnold Networks. <https://doi.org/10.48550/arXiv.2005.14165>

Open AI. (2020). Language Models are Few-Shot Learners. _arxiv.org_. <https://doi.org/10.48550/arXiv.2005.14165>

Sherstinsky, A. (2018). Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network. _arxiv.org_. <https://doi.org/10.48550/arXiv.1808.03314>

3Blue1Brown. (2017, October 5). _But what is a neural network? | Deep learning chapter 1_. YouTube. Retrieved December 26, 2024, from <https://www.youtube.com/watch?v=aircAruvnKk>

3Blue1Brown. (2017, October 16). _Gradient descent, how neural networks learn | DL2_. YouTube. Retrieved December 26, 2024, from <https://www.youtube.com/watch?v=IHZwWFHWa-w>

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. _arxiv.org_. <https://doi.org/10.48550/arXiv.1706.03762>

# Current Projects:
- GPT Architecture Language Model

## GPT Architecture Language Model
### How to Use
Get script documentation by running the following command
`python lm_model.py -h`
### Training Configurations
#### Low Performance / Test Config
1. `cd gpt`
2. `python lm_model.py -t tiktoken -m o200k_base -l <model path> -s <model path> -d <training data path> 8 8 4 1e-3 5000 4 3 0.1`
#### Current Chimera Config
1. `cd gpt`
2. `python lm_model.py -t tiktoken -m o200k_base -l <model path> -s <model path> -d <training data path> 384 64 256 3e-4 5000 6 6 0.2`
