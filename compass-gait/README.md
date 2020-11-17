# Compass gait case study

This repository contains the code necessary for reproducing the experiment concerning the compass gait walker in our recent paper **Learning Hybrid Control Barrier Functions From Data**, which appeared at CORL 2020.  If you find this code useful in your own research, please consider citing:

```latex
@article{lindemann2020learning,
  title={Learning Hybrid Control Barrier Functions from Data},
  author={Lindemann, Lars and Hu, Haimin and Robey, Alexander and Zhang, Hanwen and Dimarogonas, Dimos V and Tu, Stephen and Matni, Nikolai},
  journal={arXiv preprint arXiv:2011.04112},
  year={2020}
}
```

## Requirements

This program requires Python >= 3.7 and [Jax](https://github.com/google/jax); all experiments for the compass gait walker were run using the [Jax build with GPU support](https://github.com/google/jax#pip-installation).  Further, this repository requires the requirements in the `requirements.txt` file.  To build a virtual environment containing these dependencies, one option is to use conda:

```bash
conda create -n hcbfenv python=3.7
conda activate hcbfenv
pip3 install -r requirements.txt
```

Note that Jax is not included in `requirements.txt` to allow users to download the relevant builds.

## Collecting expert trajectories

To collect expert trajectories using the compass gait walker, you can run the following bash script:

```bash
chmod +x collect_data.sh
./collect_data.sh
```