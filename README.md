# nanoGPT

This project aims to build a nanoGPT from scratch, using the 'Shakespeare" poem dataset. The tokenizing process involves breaking words into characters. There are a total of 65 characters. Let's see how good this model can generate a good poem as Shakespeare or not!!

# Code walkthrough

## Important Notation
1. B: Batch size
2. T: Timestep, block_size (# tokens in 1 input)
3. C: Channel, embedding dimension - n_embd
4. head_size: dimensions (number of channels or features) in each head

## gpt.py
### Head - Single Head Attention
1. Masked Attention Score Matrix
2. input: (B, T, C) ---> output: (B, T, head_size)


### MultiHeadAttention
1. Perform the attention function in parallel in each child head then concatenated the result of each child head
2. input: (B, T, C) ---> n heads * (B, T, C/n) ---> output: (B, T, C)

### FeedForward
1. Linear layers that allow each token to communicate with each other after being concatenate from performing self-attention of each child head
2. input: (B, T, C) ---> (B, T, 4 * C) ---> output: (B, T, C)

### TransformerDecoderBlock
1. Perform MultiHeadAttention -> FeedForward -> Layer Normal  and Residual Connection
2. input: (B,T,C) ---> (B,T,C)

### GPTLanguageModel
1. Perform Embedding + Positional Embedding 
---> n_layer * TransformerDecoderBlock
---> Linear layer that maps (B,T,C) ---> (B,T,vocab_size)
---> Softmax: (B,T,vocab_size) ----> (B,T,1): prediction of next words given current word


## train.py
### Token:
1. Tokenizing on the character level
### Batch:
1. shuffle and get random data points of batch size

