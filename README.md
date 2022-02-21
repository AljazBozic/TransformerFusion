# TransformerFusion: Monocular RGB Scene Reconstruction using Transformers
### [Project Page](https://aljazbozic.github.io/transformerfusion) | [Paper](https://arxiv.org/pdf/2107.02191.pdf) | [Video](https://www.youtube.com/watch?v=LIpTKYfKSqw)
<br/>

> TransformerFusion: Monocular RGB Scene Reconstruction using Transformers  
> [Aljaz Bozic](https://aljazbozic.github.io), [Pablo Palafox](https://pablopalafox.github.io), [Justus Thies](https://justusthies.github.io), [Angela Dai](https://www.3dunderstanding.org/index.html), [Matthias Niessner](https://www.niessnerlab.org)  
> NeurIPS 2021

![demo](assets/transformerfusion-demo.gif)

<br/>


## TODOs
- [x] Evaluation code and metrics (with ground truth data)
- [ ] Model code (with pretrained checkpoint)
- [ ] Test-time reconstruction code
- [ ] Training (and evaluation) data preparation scripts


## How to install the framework

- Clone the repository with submodules:
```
git clone --recurse-submodules https://github.com/AljazBozic/TransformerFusion.git
```

- Create Conda environment:
```
conda env create -f environment.yml
```

- Compile local C++/CUDA dependencies:
```
conda activate tf
cd csrc
python setup.py install
```


## Evaluate the reconstructions

We evaluate method performance on the test scenes of [ScanNet dataset](http://www.scan-net.org). 

We compare scene reconstructions to the ground truth meshes, obtained with fusion of RGB-D data. Since the ground truth meshes are not complete, we additionally compute occlusion masks of RGB-D scans, to not penalize the reconstructions that are more complete than the ground truth meshes. 

You can download both ground truth meshes and occlusion masks [here](https://drive.google.com/file/d/1-nto65_JTNs1vyeHycebidYFyQvE6kt4/view?usp=sharing). To evaluate the reconstructions, you need to place them into `data/reconstructions`, and extract the ground truth data to `data/groundtruth`. The reconstructions are expected to be named as ScanNet test scenes, e.g. `scene0733_00.ply`. The following script computes evaluation metrics over all provided scene meshes:

```
conda activate tf
python src/evaluation/eval.py
```


## Citation
If you find our work useful in your research, please consider citing:

	@article{
    bozic2021transformerfusion,
    title={TransformerFusion: Monocular RGB Scene Reconstruction using Transformers},
    author={Bozic, Aljaz and Palafox, Pablo and Thies, Justus and Dai, Angela and Niessner, Matthias},
    journal={Proc. Neural Information Processing Systems (NeurIPS)},
    year={2021}}       

    

## Related work
Some other related work on monocular RGB reconstruction of indoor scenes:
* [Sun et al. - NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video (2021)](https://github.com/zju3dv/NeuralRecon)
* [Murez et al. - ATLAS: End-to-End 3D Scene Reconstruction from Posed Images (2020)](https://github.com/magicleap/Atlas)


## License

The code from this repository is released under the MIT license.