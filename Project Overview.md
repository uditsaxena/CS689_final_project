## Project Overview

### Exploring Video to Text: Generating captions of videos using RNNs  

##### Problem Statement:
Given a video - a youtube clip, a movie - generate an accurate description for the video. This description might be one sentence(captions) or multiple sentences (paragraph)

##### Previous Work:
1. Hierarchical RNNs have been used to generate paragraphs, but have problems when detecting small objects, generating nouns etc.
2. Translating Videos to Natural Language Using Deep Recurrent Neural Networks - learns the latent meaning state and a fluent grammatical model for the sentences. Falls short in better utilizing the temporal information in videos.
3. Sequence to sequence : Video to text generation - learns temporal structure of the videos. Needs work on incorporating large number of sequences in a single video format.

##### Proposed Method:
This problem is analogous to machine translation between natural languages, where a sequence of words in the input language is translated to a sequence of words in the output language. Recently work has shown to effectively attack this sequence to sequence problem with an LSTM Recurrent Neural Network (RNN). We extend this paradigm to inputs comprised of sequences of video frames

We break our proposed model into the following stages:
- Generating features from video frames. This is done using CNNs, the outputs of which are then fed into the next stage. Currently, CNNs are used on the entire image set at once - meaning only one large feature space is produced. This can be extended to applying CNNs on video subsequences to generate features for sub-actions. Pre-trained CNNs like a few from Caffe ModelZoo.
- Model the dependency between frames by using attention models like stacked LSTMs and generate intermediate features for these interdependent frames. This can be done using the first layer of the stacked LSTM.
- Process/encode these intermediate features to generate text based features.
- Use these text based features to generate an accurate description of the video by generating one or more than one sentence using deeper layers of the LSTM. 

##### Data Sets:
Some datasets we would want to use are:
- Microsoft Video Description Corpus - contains Youtube video clips with annotated text description.
- MPII Movie Description Corpus
-  Montreal Video Annotation Dataset

##### Evaluation Metrics:
We hope to evaluate sentence models by using the following metrics:
- BLEU (bilingual evaluation understudy): evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is".
- METEOR: an automatic metric for machine translation evaluation that is based on a generalized concept of unigram matching between the machine produced translation and human-produced reference translations.


##### Proposed Timeline:
We plan to proceed over the four weeks as follows:
1. Week 1 - Literature survey of related work and setting up basic code and model architecture for future work.
2. Week 2 - Implementing attention based model architecture.
3. Week 3 - Evaluation of metrics and working on improving results by tweaking hyper-parameters of our model
4. Week 4 - Analyzing results and submitting review document.

##### Challenges to tackle:
- Modeling frame interdependence within the video.
- Implementing effective attention based architecture.
- Implementing an effective text description (sentence/paragraph) architecture.
