# Conditional GANs with Auxiliary Discriminative Classifier
> Liang Hou, Qi Cao, Huawei Shen, Siyuan Pan, Xiaoshuang Li, Xueqi Cheng
>
> International Conference on Machine Learning (ICML), 2022

This is a PyTorch implementation of [Conditional GANs with Auxiliary Discriminative Classifier](https://arxiv.org/abs/2107.10060) (ADC-GAN) based on the [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) repository. We note that the [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) repository also provides an implementation of our ADC-GAN, which facilitates fair comparisons of state-of-the-art GANs. The experiments in the paper are conducted using the two repositories.


$$
\begin{align}
\max_{D,C_\mathrm{d}}V(G,D)+\lambda\cdot\left(\mathbb{E}\_{x,y\sim p(x,y)}[\log C_\mathrm{d}(y^+|x)]+\mathbb{E}\_{x,y\sim q(x,y)}[\log C_\mathrm{d}(y^-|x)]\right)
\\
\min_{G}V(G,D)-\lambda\cdot\left(\mathbb{E}\_{x,y\sim q(x,y)}[\log C_\mathrm{d}(y^+|x)]-\mathbb{E}\_{x,y\sim q(x,y)}[\log C_\mathrm{d}(y^-|x)]\right) \\
\end{align}
$$


If you find our work useful, please consider citing our paper:
```

@InProceedings{pmlr-v162-hou22a,
  title = 	 {Conditional {GAN}s with Auxiliary Discriminative Classifier},
  author =       {Hou, Liang and Cao, Qi and Shen, Huawei and Pan, Siyuan and Li, Xiaoshuang and Cheng, Xueqi},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {8888--8902},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/hou22a/hou22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/hou22a.html},
  abstract = 	 {Conditional generative models aim to learn the underlying joint distribution of data and labels to achieve conditional data generation. Among them, the auxiliary classifier generative adversarial network (AC-GAN) has been widely used, but suffers from the problem of low intra-class diversity of the generated samples. The fundamental reason pointed out in this paper is that the classifier of AC-GAN is generator-agnostic, which therefore cannot provide informative guidance for the generator to approach the joint distribution, resulting in a minimization of the conditional entropy that decreases the intra-class diversity. Motivated by this understanding, we propose a novel conditional GAN with an auxiliary discriminative classifier (ADC-GAN) to resolve the above problem. Specifically, the proposed auxiliary discriminative classifier becomes generator-aware by recognizing the class-labels of the real data and the generated data discriminatively. Our theoretical analysis reveals that the generator can faithfully learn the joint distribution even without the original discriminator, making the proposed ADC-GAN robust to the value of the coefficient hyperparameter and the selection of the GAN loss, and stable during training. Extensive experimental results on synthetic and real-world datasets demonstrate the superiority of ADC-GAN in conditional generative modeling compared to state-of-the-art classifier-based and projection-based conditional GANs.}
}
```
