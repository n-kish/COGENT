# COGENT: Co-Design of Robots with Generative Flow Networks

<!-- <p align="center">
·
<a href="[Link to your paper/publication]">Paper</a>
·
<a href="[Link to your GitHub repository]">Code</a>
·
<a href="[Link to your project website]">Website</a>
·
<a href="[Link to your Hugging Face model/dataset]">Hugging Face</a>
</p> -->

This repository contains the implementation of our research work "COGENT".
<!-- ([Conference/Journal Name Year], [Publication Type]). [Additional information about presentation/publication]. -->

COGENT is a novel framework that leverages the graph synthesis technique of GFlowNets to enhance search space traversal in robotic co-design. To increase sample efficiency, the proposed framework introduces a cost/performance-aware design prioritization mechanism that learns a design generator policy by carefully sampling the design space.

<p align="center">
    <br>
    <img src="figures/COGENT_methodology.png"/>
    <br>
<p>

COGENT framework can be broadly divided into two main stages, as depicted in the flow diagram:

<h>Generator
The Generator is responsible for proposing a batch of 'm' diverse robot designs, each represented as a graph. Starting from initial states, the generator incrementally constructs each of the m graphs by iteratively applying Graph Actions (AddNode, AddEdge, STOP) selected from a defined action space. These actions are guided by the learned forward policy, which is influenced by our annealing exploration strategy. The process continues independently for each graph in the batch until a STOP action is sampled, yielding m complete robot design graphs.

<h>Evaluator
The Evaluator takes the batch of m generated robot design graphs and assesses their potential task performance. Firstly, these graph representations are converted into a format suitable for simulation, such as XML, representing the robot's structure. These robots are then instantiated within a physics simulator, where their behavior policies are trained using an policy optimization algorithm (like PPO) for a specified number of timesteps. The performance achieved during this training is then processed by our proposed techniques: Rate-based Prioritization and Cost-Aware Sampling. These techniques modify the raw simulation reward to produce a composite reward that incorporates factors beyond just final performance, reflecting the potential for improvement and the computational cost of evaluation. This composite reward is then used to update the Generator's design policy and the partition function via the TB objective, closing the loop.

## 🛠️ Setup
Let's start with python 3.10. It's recommend to create a `conda` env:

### Create a new conda environment 
```bash
conda create -n cogent python=3.10
conda activate cogent
```

### Install dependencies
```bash
pip install -r requirements.txt
```

<!-- ### (Optional) Pretrained Models
We provide pretrained models in `[path/to/models]` for visualization.

* You can download pretrained models from [Download Link]

* [Instructions for using the pretrained models]

## 💻 Training
```bash
[Training command/script]
```

[Detailed explanation of training process, hyperparameters, etc.] -->

## 🙏 Acknowledgement
<p>Many thanks to the [<a href="https://github.com/recursionpharma" target="_blank">Recursion Pharma</a>] repository and [<a href="https://github.com/bengioe" target="_blank">Yoshua Bengio's Lab</a>] for making GFlowNet accessible to everyone.</p>

<!-- ## 📚 Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{
  [citation information]
}
``` -->

## 🏷️ License
Please see the [license](LICENSE) for further details.
