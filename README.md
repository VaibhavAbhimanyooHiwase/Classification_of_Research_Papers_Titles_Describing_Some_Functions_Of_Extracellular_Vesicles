# Classifying_Research_Papers_Based_On_Whether_They_Describe_Some_Functions_Of_Extracellular_Vesicles
Here the main topics between two classes are similar (both mentioning EVs), they should be classified using other clues. Most classifications use simple clues, e.g. words like single words (1-gram), biwords (2-gram), triwords (3-gram). People need a classifier which recognize words that are likely to describe a biological function, such as "prevent cardiomyocytes apoptosis". This means that it may need a large amount of training data, which is limited due to limited the number of papers, published on EVs. We proposed an approach that use the keywords of research paper as feature and generate a Restricted Boltzmann Machine (RBM). This help us to generate the extracellular vesicles (EVs) related concept in the hidden nodes, when training of n-gram words generated from paper titles are given to the visible nodes. By adjusting weight using sigmoid activation function, we minimize errors for Bernoulli classification. We use contrastive divergence algorithm to reduce steps of Markov chain Monte Carlo methods in Gibbs Sampling. Final score is generated by normalizing number of keyword matched by classifier. The 0.5 threshold value of this normalized score help us to generate binary classification (belonging to EVs / not belonging to EVs) with precision above 80% for other author’s analysis. Keywords of research papers are the feature selection for this problem. Hence, if paper is not related to extracellular vesicles (EVs), their keywords are the features which add noise to the training of the model by creating other concepts along with EVs in the hidden nodes. We try to reduce this noise by extracting keywords only from the paper which are related to EVs.
