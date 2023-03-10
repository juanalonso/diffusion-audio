# Modelos de audio, aplicaciones y utilidades basadas en diffusion
### (y algún otra que me resulte interesante)

<br>

* [ArchiSound](#ArchiSound)
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
* [SingSong](#SingSong)

<br><br>

##### 2023-01-31
## AudioLDM
#### 16kHz

This study proposes AudioLDM, a text-to-audio system that is built on a latent space to learn continuous audio representations from language-audio pretraining latents and enables various text-guided audio manipulations. AudioLDM is advantageous in both generation quality and computational efficiency, and achieves state-of-the-art performance when trained on AudioCaps with a single GPU.

[Paper](https://arxiv.org/abs/2301.12503)
–
[Demo](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)
–
[Examples](https://audioldm.github.io/)

<br><br>

##### 2023-01-30
## SingSong
#### 44kHz

SingSong is a system which generates instrumental music to accompany input vocals, and uses recent developments in musical source separation and audio generation. Through source separation, aligned pairs of vocals and instrumentals are produced, and AudioLM is adapted for conditional audio-to-audio generation tasks. Tests show improved performance on isolated vocals by 53%, and listeners expressed preference for instrumentals generated by SingSong compared to those from a retrieval baseline.

[Paper](https://arxiv.org/abs/2301.11757)
–
[Examples](https://storage.googleapis.com/sing-song/index.html?s=35c)

<br><br>

##### 2023-01-30
## Moûsai
#### 48kHz

This work investigates the potential of text-conditional music generation using a cascading latent diffusion approach which can generate high-quality music at 48kHz from textual descriptions, while maintaining reasonable inference speed on a single consumer GPU.

[Paper](https://arxiv.org/abs/2301.11757)
–
[Code](https://github.com/archinetai/audio-diffusion-pytorch)
 –
[Examples](https://anonymous0.notion.site/anonymous0/Mo-sai-Text-to-Audio-with-Long-Context-Latent-Diffusion-b43dbc71caf94b5898f9e8de714ab5dc)

<br><br>

##### 2023-01-30
## Archisound
#### 48kHz

Diffusion models have become increasingly popular for image generation, and this has sparked interest in their potential applications to audio generation. This work explores the potential of diffusion models for audio generation by proposing a set of models that address multiple aspects such as temporal dimension, long term structure, and overlapping sounds. In order to maintain reasonable inference speed, these models are designed to achieve real-time results on a single consumer GPU. Open source libraries are also provided to facilitate further research in the field.

[Paper](https://arxiv.org/abs/2301.13267)
–
[Code](https://github.com/archinetai/audio-diffusion-pytorch)
 –
[Examples](https://flavioschneider.notion.site/flavioschneider/Audio-Generation-with-Diffusion-c4f29f39048d4f03a23da13078a44cdb)

<br><br>


##### 2023-01-29
## Make-An-Audio
#### 16kHz

Make-An-Audio is a prompt-enhanced diffusion model for large-scale text-to-audio generation. It alleviates data scarcity with pseudo prompt enhancement and leverages spectrogram autoencoder to predict the self-supervised audio representation, achieving state-of-the-art results in objective and subjective evaluations. We also present its controllability with classifier-free guidance and generalization for X-to-Audio with "No Modality Left Behind".

[Paper](https://text-to-audio.github.io/paper.pdf)
–
[Examples](https://text-to-audio.github.io/)

<br><br>

##### 2023-01-28
## Msanii
#### 44kHz

Msanii, a novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently. This is the first work to successfully employ diffusion models for synthesizing such long music samples at high sample rates.

[Paper](https://arxiv.org/abs/2301.06468)
–
[Code](https://github.com/Kinyugo/msanii)
–
[Demo](https://huggingface.co/spaces/kinyugo/msanii)
–
[Examples](https://kinyugo.github.io/msanii-demo/)

<br><br>

##### 2023-01-28
## Noise2Music

We introduce Noise2Music, where a series of diffusion models is trained to generate high-quality 30-second music clips from text prompts. Two types of diffusion models are trained and utilized in succession to generate high-fidelity music.

We explore two options for the intermediate representation, one using a spectrogram and the other using audio with lower fidelity. The generated audio faithfully reflects key elements of the text prompt such as genre, tempo, instruments, mood and era, but goes beyond to ground fine-grained semantics of the prompt. 

[Examples](https://noise2music.github.io/)

<br><br>

##### 2023-01-27
## RAVE 2
#### 48kHz

This paper introduces a real-time Audio Variational autoEncoder (RAVE) for fast and high-quality audio waveform synthesis. We show that it is the first model able to generate 48kHz audio signals while running 20 times faster than real-time on a standard laptop CPU. Our novel two-stage training procedure and post-training analysis of the latent space allows direct control over reconstruction fidelity and representation compactness. We evaluate the quality of the synthesized audio using quantitative and qualitative experiments, showing its superiority over existing models. Finally, we demonstrate applications of our model for timbre transfer and signal compression. All of our source code and audio examples are publicly available. 

[Paper](https://arxiv.org/abs/2111.05011)
–
[Code](https://github.com/acids-ircam/RAVE)
–
[Demo](https://www.youtube.com/watch?v=jAIRf4nGgYI)
–
[Examples](https://rmfrancis.bandcamp.com/album/pedimos-un-mensaje)

<br><br>

### 2023-01-27
## MusicLM
##### 24kHz

We introduce MusicLM, a model that generates high-fidelity music from text descriptions. We also release MusicCaps, a dataset of 5.5k music-text pairs with rich text descriptions provided by human experts. Our experiments show that MusicLM outperforms previous systems both in audio quality and adherence to the text description, and it can be conditioned on both text and a melody in that it can transform whistled and hummed melodies according to the style described in a text caption. 

[Paper](https://arxiv.org/abs/2301.11325)
–
[3rd. party code](https://github.com/lucidrains/musiclm-pytorch)
–
[Examples](https://google-research.github.io/seanet/musiclm/examples/)

<br><br>

##### 2022-11-25
## Riffusion

 Riffusion is a library for real-time music and audio generation with stable diffusion.

[Paper](https://www.riffusion.com/about)
–
[Code](https://github.com/riffusion/riffusion)
–
[Demo](https://www.riffusion.com/)

<br><br>

##### 2022-09-07
## AudioLM

AudioLM is a framework for high-quality audio generation with long-term consistency. It maps the input audio to a sequence of discrete tokens and casts audio generation as a language modeling task. A hybrid tokenization scheme is proposed to achieve both reconstruction quality and long-term structure. AudioLM has been used to generate speech continuations while maintaining speaker identity and prosody, as well as piano music continuations without any symbolic representation of music.

[Paper](https://arxiv.org/abs/2209.03143)
–
[3rd. party code](https://github.com/lucidrains/audiolm-pytorch)
–
[Examples](https://google-research.github.io/seanet/audiolm/examples/)
–
[Extra info](https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html)

<br><br>

##### 2022-07-22
## Dance Diffusion
#### 16-44kHz

[Code](https://github.com/harmonai-org/sample-generator)
–
[Demo](https://colab.research.google.com/github/harmonai-org/sample-generator/blob/main/Dance_Diffusion.ipynb)

<br><br>

##### 2020-09-21
## DiffWave
##### 22kHz

This paper introduces DiffWave, a non-autoregressive, diffusion probabilistic model for waveform generation. It is more efficient than WaveNet vocoders and outperforms autoregressive and GAN-based models in terms of audio quality and sample diversity.

[Paper](https://arxiv.org/abs/2009.09761)
–
[Code](https://github.com/lmnt-com/diffwave)
–
[3rd. party code](https://github.com/philsyn/DiffWave-Vocoder)
–
[Examples](https://diffwave-demo.github.io/)