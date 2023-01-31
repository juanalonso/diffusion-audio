# Modelos de audio, aplicaciones y utilidades basadas en diffusion
### (y algún otra que me resulte interesante)

<br>

* [AudioLDM](#AudioLDM)
* [AudioLM](#AudioLM)
* [Dance Diffusion](#Dance-Diffusion)
* [DiffWave](#DiffWave)
* [Make-An-Audio](#Make-An-Audio)
* [Moûsai](#Moûsai)
* [Msanii](#Msanii)
* [MusicLM](#MusicLM)
* [Noise2Music](#Noise2Music)
* [RAVE 2](#RAVE-2)
* [Riffusion](#Riffusion)

<br> <br> <br>

#### 2023-01-31
## AudioLDM
###  Haohe Liu, Zehua Chen, Yi Yuan1, Xinhao Mei1, Xubo Liu, Danilo Mandic, Wenwu Wang, Mark D. Plumley 

This study proposes AudioLDM, a text-to-audio system that is built on a latent space to learn continuous audio representations from language-audio pretraining latents and enables various text-guided audio manipulations. AudioLDM is advantageous in both generation quality and computational efficiency, and achieves state-of-the-art performance when trained on AudioCaps with a single GPU.

| | |
|-|-|
| Paper | https://arxiv.org/abs/2301.12503 |
| Ejemplos | https://audioldm.github.io/ |
| Demo | No |
| Código oficial | No |
| Código de terceros | No |
| Output | 16kHz |

<br>

---
<br> <br>




#### 2023-01-30
## Moûsai
### Flavio Schneider, Zhijing Jin, Bernhard Schölkopf

 This work investigates the potential of text-conditional music generation using a cascading latent diffusion approach which can generate high-quality music at 48kHz from textual descriptions, while maintaining reasonable inference speed on a single consumer GPU.

| | |
|-|-|
| Paper | https://arxiv.org/abs/2301.11757 |
| Ejemplos | https://anonymous0.notion.site/anonymous0/Mo-sai-Text-to-Audio-with-Long-Context-Latent-Diffusion-b43dbc71caf94b5898f9e8de714ab5dc |
| Demo | No |
| Código oficial | https://github.com/archinetai/audio-diffusion-pytorch |
| Código de terceros | No |
| Output | 48kHz |

<br>

---
<br> <br>


#### 2023-01-29
## Make-An-Audio
### Zhejiang University

Make-An-Audio is a prompt-enhanced diffusion model for large-scale text-to-audio generation. It alleviates data scarcity with pseudo prompt enhancement and leverages spectrogram autoencoder to predict the self-supervised audio representation, achieving state-of-the-art results in objective and subjective evaluations. We also present its controllability with classifier-free guidance and generalization for X-to-Audio with "No Modality Left Behind".

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
## Msanii
### Kinyugo Maina

 Msanii, a novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently. This is the first work to successfully employ diffusion models for synthesizing such long music samples at high sample rates.

| | |
|-|-|
| Paper | https://arxiv.org/abs/2301.06468 |
| Ejemplos | https://kinyugo.github.io/msanii-demo/ |
| Demo | https://huggingface.co/spaces/kinyugo/msanii |
| Código oficial | https://github.com/Kinyugo/msanii |
| Código de terceros | No |
| Output | 44kHz |

<br>

---
<br> <br>

#### 2023-01-28
## Noise2Music
### Anonymous

We introduce Noise2Music, where a series of diffusion models is trained to generate high-quality 30-second music clips from text prompts. Two types of diffusion models are trained and utilized in succession to generate high-fidelity music.

We explore two options for the intermediate representation, one using a spectrogram and the other using audio with lower fidelity. The generated audio faithfully reflects key elements of the text prompt such as genre, tempo, instruments, mood and era, but goes beyond to ground fine-grained semantics of the prompt. 

| | |
|-|-|
| Paper | No |
| Ejemplos | https://noise2music.github.io/ |
| Demo | No |
| Código oficial | No |
| Código de terceros | No |
| Output | ?kHz |

<br>

---
<br> <br>


#### 2023-01-27
## RAVE 2
### ACIDS IRCAM

This paper introduces a real-time Audio Variational autoEncoder (RAVE) for fast and high-quality audio waveform synthesis. We show that it is the first model able to generate 48kHz audio signals while running 20 times faster than real-time on a standard laptop CPU. Our novel two-stage training procedure and post-training analysis of the latent space allows direct control over reconstruction fidelity and representation compactness. We evaluate the quality of the synthesized audio using quantitative and qualitative experiments, showing its superiority over existing models. Finally, we demonstrate applications of our model for timbre transfer and signal compression. All of our source code and audio examples are publicly available. 

| | |
|-|-|
| Paper | https://arxiv.org/abs/2111.05011 |
| Ejemplos | https://rmfrancis.bandcamp.com/album/pedimos-un-mensaje |
| Demo | https://www.youtube.com/watch?v=jAIRf4nGgYI |
| Código oficial | https://github.com/acids-ircam/RAVE |
| Código de terceros | No |
| Output | 48kHz |

<br>

---
<br> <br>

### 2023-01-27
## MusicLM
### Google

We introduce MusicLM, a model that generates high-fidelity music from text descriptions. We also release MusicCaps, a dataset of 5.5k music-text pairs with rich text descriptions provided by human experts. Our experiments show that MusicLM outperforms previous systems both in audio quality and adherence to the text description, and it can be conditioned on both text and a melody in that it can transform whistled and hummed melodies according to the style described in a text caption. 

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
## Riffusion
### Seth Forsgren y Hayk Martiros

 Riffusion is a library for real-time music and audio generation with stable diffusion.

| | |
|-|-|
| Paper | https://www.riffusion.com/about |
| Ejemplos | No |
| Demo | https://www.riffusion.com/ |
| Código oficial | https://github.com/riffusion/riffusion |
| Código de terceros | No |
| Output | kHz |

<br>

---
<br> <br>

#### 2022-09-07
## AudioLM
### Google

AudioLM is a framework for high-quality audio generation with long-term consistency. It maps the input audio to a sequence of discrete tokens and casts audio generation as a language modeling task. A hybrid tokenization scheme is proposed to achieve both reconstruction quality and long-term structure. AudioLM has been used to generate speech continuations while maintaining speaker identity and prosody, as well as piano music continuations without any symbolic representation of music.

| | |
|-|-|
| Paper | https://arxiv.org/abs/2209.03143 |
| Info. adicional | https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html |
| Ejemplos | https://google-research.github.io/seanet/audiolm/examples/ |
| Demo | No |
| Código oficial | No |
| Código de terceros | https://github.com/lucidrains/audiolm-pytorch |
| Destacado | Preserve speaker identity, prosody, accent and recording conditions of the prompt  |
| Output | ?kHz |

<br>

---
<br> <br>


#### 2022-07-22
## Dance Diffusion
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

<br>

---
<br> <br>

#### 2020-09-21
## DiffWave
### Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro

 This paper introduces DiffWave, a non-autoregressive, diffusion probabilistic model for waveform generation. It is more efficient than WaveNet vocoders and outperforms autoregressive and GAN-based models in terms of audio quality and sample diversity.

| | |
|-|-|
| Paper | https://arxiv.org/abs/2009.09761 |
| Ejemplos | https://diffwave-demo.github.io/ |
| Demo | No |
| Código oficial | https://github.com/lmnt-com/diffwave |
| Código de terceros | https://github.com/philsyn/DiffWave-Vocoder |
| Output | 22kHz |

