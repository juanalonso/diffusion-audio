# Modelos de audio, aplicaciones y utilidades basadas en diffusion
### (y algún otra que me resulte interesante)

<br>

* [Make-An-Audio](#Make-An-Audio)
* [Noise2Music](#Noise2Music)
* [RAVE 2](#RAVE2)
* [MusicLM](#MusicLM)
* [Riffusion](#Riffusion)
* [AudioLM](#AudioLM)
* [Dance Diffusion](#DanceDiffusion)

<br> <br> <br>




#### 2023-01-29
## <a name="Make-An-Audio"></a>Make-An-Audio
### Zhejiang University

 Large-scale multimodal generative modeling has created milestones in text-to-image and text-to-video generation. Its application to audio still lags behind due to two main reasons: the lack of large-scale datasets with high-quality text-audio pairs, and the complexity of modeling long continuous audio data. In this work, we propose Make-An-Audio with a prompt-enhanced diffusion model that addresses these gaps by 1) introducing pseudo prompt enhancement with a distill-then-reprogram approach which alleviates the data scarcity by using weekly-supervised data with language-free audios; 2) leveraging spectrogram autoencoder to predict the self-supervised audio representation instead of waveforms. Together with robust contrastive language-audio pretraining (CLAP) representations, Make-An-Audio achieves state-of-the-art results in both objective and subjective evaluation. Moreover, we present its controllability with classifier-free guidance and generalization for X-to-Audio with "No Modality Left Behind", for the first time unlocking the ability to generate high-definition, high-fidelity audios given a user-defined modality input. 

| | |
|-|-|
| Paper | https://text-to-audio.github.io/paper.pdf |
| Ejemplos | https://text-to-audio.github.io/ |
| Demo | No |
| Código oficial | No |
| Código de terceros | No |
| Destacado | Text to audio, Image to audio, Video to audio |
| Output | 16kHz |

<br>

---
<br> <br>


#### 2023-01-28
## <a name="Noise2Music"></a>Noise2Music
### Anonymous

 We introduce Noise2Music, where a series of diffusion models is trained to generate high-quality 30-second music clips from text prompts. Two types of diffusion models, a generator model, which generates an intermediate representation conditioned on text, and a cascader model, which generates high-fidelity audio conditioned on the intermediate representation and possibly the text, are trained and utilized in succession to generate high-fidelity music.

We explore two options for the intermediate representation, one using a spectrogram and the other using audio with lower fidelity. We find that the generated audio is not only able to faithfully reflect key elements of the text prompt such as genre, tempo, instruments, mood and era, but goes beyond to ground fine-grained semantics of the prompt. Pretrained large language models play a key role in this story---they are used to generate paired text for the audio of the training set and to extract embeddings of the text prompts ingested by the diffusion models. 

| | |
|-|-|
| Paper | No |
| Ejemplos | https://noise2music.github.io/ |
| Demo | No |
| Código oficial | No |
| Código de terceros | No |
| Destacado | ? |
| Output | ?kHz |

<br>

---
<br> <br>


#### 2023-01-27
## <a name="RAVE2"></a>RAVE 2
### ACIDS IRCAM

 Deep generative models applied to audio have improved by a large margin the state-of-the-art in many speech and music related tasks. However, as raw waveform modelling remains an inherently difficult task, audio generative models are either computationally intensive, rely on low sampling rates, are complicated to control or restrict the nature of possible signals. Among those models, Variational AutoEncoders (VAE) give control over the generation by exposing latent variables, although they usually suffer from low synthesis quality. In this paper, we introduce a Realtime Audio Variational autoEncoder (RAVE) allowing both fast and high-quality audio waveform synthesis. We introduce a novel two-stage training procedure, namely representation learning and adversarial fine-tuning. We show that using a post-training analysis of the latent space allows a direct control between the reconstruction fidelity and the representation compactness. By leveraging a multi-band decomposition of the raw waveform, we show that our model is the first able to generate 48kHz audio signals, while simultaneously running 20 times faster than real-time on a standard laptop CPU. We evaluate synthesis quality using both quantitative and qualitative subjective experiments and show the superiority of our approach compared to existing models. Finally, we present applications of our model for timbre transfer and signal compression. All of our source code and audio examples are publicly available. 

| | |
|-|-|
| Paper | https://arxiv.org/abs/2111.05011 |
| Ejemplos | https://rmfrancis.bandcamp.com/album/pedimos-un-mensaje |
| Demo | https://www.youtube.com/watch?v=jAIRf4nGgYI |
| Código oficial | https://github.com/acids-ircam/RAVE |
| Código de terceros | No |
| Destacado |  |
| Output | ?kHz |

<br>

---
<br> <br>

### 2023-01-27
## <a name="MusicLM"></a>MusicLM
### Google

We introduce MusicLM, a model generating high-fidelity music from text descriptions such as "a calming violin melody backed by a distorted guitar riff". MusicLM casts the process of conditional music generation as a hierarchical sequence-to-sequence modeling task, and it generates music at 24 kHz that remains consistent over several minutes. Our experiments show that MusicLM outperforms previous systems both in audio quality and adherence to the text description. Moreover, we demonstrate that MusicLM can be conditioned on both text and a melody in that it can transform whistled and hummed melodies according to the style described in a text caption. To support future research, we publicly release MusicCaps, a dataset composed of 5.5k music-text pairs, with rich text descriptions provided by human experts. 

| | |
|-|-|
| Paper |http://arxiv.org/abs/2301.11325|
| Ejemplos | https://google-research.github.io/seanet/musiclm/examples/ |
| Demo |No|
| Código oficial | No |
| Código de terceros | https://github.com/lucidrains/musiclm-pytorch (placeholder) |
| Destacado | Text to audio (captions, genres, musician experience, places, periods in time), Long Generation, Story Mode, Text and Melody Conditioning, Painting Caption Conditioning|
| Output |24kHz|

<br>

---
<br> <br>


#### 2022-11-25
## <a name="Riffusion"></a>Riffusion
### Seth Forsgren y Hayk Martiros

 Riffusion is a library for real-time music and audio generation with stable diffusion.

| | |
|-|-|
| Paper | https://www.riffusion.com/about |
| Ejemplos | No |
| Demo | https://www.riffusion.com/ |
| Código oficial | https://github.com/riffusion/riffusion |
| Código de terceros | No |
| Destacado |  |
| Output | kHz |

<br>

---
<br> <br>

#### 2022-09-07
## <a name="AudioLM"></a>AudioLM
### Google

We introduce AudioLM, a framework for high-quality audio generation with long-term consistency. AudioLM maps the input audio to a sequence of discrete tokens and casts audio generation as a language modeling task in this representation space. We show how existing audio tokenizers provide different trade-offs between reconstruction quality and long-term structure, and we propose a hybrid tokenization scheme to achieve both objectives. Namely, we leverage the discretized activations of a masked language model pre-trained on audio to capture long-term structure and the discrete codes produced by a neural audio codec to achieve high-quality synthesis. By training on large corpora of raw audio waveforms, AudioLM learns to generate natural and coherent continuations given short prompts. When trained on speech, and without any transcript or annotation, AudioLM generates syntactically and semantically plausible speech continuations while also maintaining speaker identity and prosody for unseen speakers. Furthermore, we demonstrate how our approach extends beyond speech by generating coherent piano music continuations, despite being trained without any symbolic representation of music. 

| | |
|-|-|
| Paper | https://arxiv.org/abs/2209.03143 |
| Info. adicional | https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html |
| Ejemplos | https://google-research.github.io/seanet/audiolm/examples/ |
| Demo | No |
| Código oficial | No |
| Código de terceros | https://github.com/lucidrains/audiolm-pytorch |
| Destacado | Preserve speaker identity, prosody, accent and recording conditions of the prompt,  |
| Output | ?kHz |

<br>

---
<br> <br>


#### 2022-07-22
## <a name="DanceDiffusion"></a>Dance Diffusion
### Harmonai

| | |
|-|-|
| Paper | No |
| Ejemplos | No |
| Demo | https://colab.research.google.com/github/harmonai-org/sample-generator/blob/main/Dance_Diffusion.ipynb |
| Código oficial | https://github.com/harmonai-org/sample-generator |
| Código de terceros | No |
| Destacado | Incluye código de entrenamiento y finetuning |
| Output | 16-44kHz |
