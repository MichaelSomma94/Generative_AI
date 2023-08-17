# A simple Bigram Model

this is a training script for the simplest possible causual Languge model, a Bigram Model.

It is trained on Shakespeare Texts comming from the Gutenberg Project

Model description:

The model uses the simplest form of Tokeinzation. It takes all appearing characters in the dataset and maps them to integers. Then the whole dataset is translated into this embeding convention. 

The model consists simplis of an emebding table, a neuronal network architecture that is trained to predict transition probalilites for the next character based on the previous one.

In the BrigramModel_output.txt file you can see what the model produces after being trained
