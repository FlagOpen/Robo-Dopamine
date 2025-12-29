
<h1 align="center">Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation</h1>

<h3 align="center">Joy is dopamine’s handiwork—whether in humans or in robotics.</h3>


<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Stay%20tuned-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="https://robo-dopamine.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/🤗%20Dataset-Stay%20tuned-green.svg" alt="Benchmark"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Stay%20tuned-yellow" alt="Weights"></a>
</p>


<div style="text-align: center; background-color: white;">
    <img src="assets/teasor.png" width=100% >
</div>


## 🗞️ News
- **`2025-12-30`**: ✨ *Codes, Dataset and Weights are coming soon!* *Stay tuned for updates*.
- **`2025-12-30`**: 🔥 We released our [Project Page](https://robo-dopamine.github.io/) of **Robo-Dopamine**.


## 🎯 TODO
- [ ] Release the 3B Dopamine-Reward (GRM) model and inference codes *(About 2 week)*.
- [ ] Release the 8B Dopamine-Reward (GRM) model *(About 2 week)*.
- [ ] Release the Robo-Dopamine Dataset and SFT training code *(About 1 months)*.
- [ ] Release the Dataset Generation Pipeline *(Maybe 1 months or more)*.
- [ ] Release the Dopamine-RL training code for simulator and real-world settings *(Maybe 2 months or more)*.


## 🤖 Overview

**Robo-Dopamine** is composed of two core components: **(a) Dopamine-Reward Modeling Method** and **(b) Dopamine-RL Training Framework**. 
**(a)** At the heart of our reward modeling is to build the General Reward Model (GRM), a vision-language model that is prompted with a task description and conditioned on multi-view images of initial, 
goal, "BEFORE," and "AFTER" states to predict a relative progress or regress hop. To ensure a stable and accurate signal, we employ **Multi-Perspective Progress Fusion**, which combines incremental, 
forward-anchored, and backward-anchored predictions into a final fused reward. 
**(b)** The Dopamine-RL framework first adapts the pre-trained GRM to a novel task using a single demonstration, i.e., **One-Shot GRM Adaptation**. 
Subsequently, it uses a theoretically-sound **Policy-Invariant Reward Shaping** method to convert the GRM's dense output into a reward signal that accelerates learning without altering the optimal policy. 
This approach is universally compatible with a wide range of RL algorithms.

<div align="center"> 
    <img src="assets/method.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>


## 📑 Citation

If you find our work helpful, feel free to cite it:
```
@article{tan2025robodopamine,
    title={Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation}, 
    author={Tan, Huajie and Chen, Sixiang and Xu, Yijie and Wang, Zixiao and Ji, Yuheng and Chi, Cheng and Lyu, Yaoxu and Zhao, Zhongxia and Chen, Xiansheng and Co, Peterson and Xie, Shaoxuan and Yao, Guocai and Wang, Pengwei and Wang, Zhongyuan and Zhang, Shanghang},
    journal={TODO},
    year={2025}
}
```