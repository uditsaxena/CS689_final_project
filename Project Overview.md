## Project Overview

### Exploring Video to Text: Generating captions of videos using RNNs  

##### Problem Statement:
Given a video - a youtube clip, a movie - generate an accurate description for the video. This description might be one sentence(captions) or multiple sentences (paragraph)

##### Previous Work:
1. Hierarchicial RNNs have been used to generate paragraphs, but have problems when detecting small objects, generating nouns etc.
2. Translating Videos to Natural Language Using Deep Recurrent Neural Networks - learns the latent meaning state and a fluent grammatical model for the sentences. Falls short in better utilizing the temporal information in videos.
3. Sequence to sequence : Video to text generation - learns temporal structure of the videos. Needs work on incorporating large number of sequences in a single video format.

##### Proposed Method:
We break our proposed model into the following stages:
- Generating features from video frames. These features are then fed into the next stage.
- Model the dependency between frames by using attention models like stacked LSTMs and generate intermediate features for these interdependent frames.
- Process/encode these intermediate features to generate text based features.
- Use these text based features to generate an accurate description of the video by generating one or more than one sentence.

##### Evaluation Metrics:
We hope to evaluate sentence models by using the following metrics:
- BLEU (bilingual evaluation understudy): evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is".
- METEOR: an automatic metric for machine translation evaluation that is based on a generalized concept of unigram matching between the machine produced translation and human-produced reference translations.


##### Proposed Timeline:


##### Challenges to tackle:
