# A GPT model

This is a GPT Model that was trained on the Gutenberg Project Shakespeare text data.

It follows the instruction of the Video from Anrej Karpathy (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=13s) which I recommend you to have a look at. It somehow leads you the way thorugh the evolution of NLP over the last decade or so. But If you just want to train a model go straight ahead!

The Model incorporates the main features of the Paper "Attention is all you need!" from 2017. So it has postional encoding of the token in the sentence and the self Attnention Mechnism. So the current token's query vector is "dot-producted" with the key vectors of the last elements, where block_size is the context length, called affinities. It also uses the batch normaliztion of the affinites used to calculate the logits (probailites for the next token) 
