
### Q-RoFormer: Quaternion Rotation-based Transformer for cross-subject EEG emotion recognition
##### Core idea: How to represent EEG with quaternion + Improving cross-subject generalization of EEG using quaternion rotation.

### News
#### 🎉🎉🎉 This paper is currently under review by [TAC](https://www.computer.org/csdl/journal/ta)

## Abstract
![Network Architecture](/fig2.png)
Electroencephalography (EEG)-based emotion recognition holds significant potential in affective brain–computer interfaces. However, substantial inter-subject variability—due to differences in skull anatomy, electrode placement, and emotional responses—leads to severe spatiotemporal distribution shifts, limiting model generalizability. To address this issue, this paper proposes a hypercomplex neural network model based on quaternion rotation, termed Q-RoFormer, which explicitly addresses cross-subject spatiotemporal distribution shifts from both spatial and temporal perspectives. Specifically, the quaternion Transformer module embeds multi-temporal-window EEG segments into a unified quaternion representation, leverages inter-window correlations in quaternion space to mitigate temporal distribution shifts, and applies quaternion rotations to align spatial distribution shifts across subjects. Subsequently, the quaternion long short-term memory network captures temporal dependencies across windows within the quaternion-valued sequence. Furthermore, Q-RoFormer adopts a pre-training–fine-tuning paradigm to bolster transfer learning capabilities. Extensive experiments on the SEED and SEED-IV datasets demonstrate the effectiveness of Q-RoFormer, achieving 93.47\% and 81.92\% accuracy, respectively. Ablation results confirm the role of spatial rotation in cross-subject generalization. 

# In short, we did three things:
- We implemented the use of quaternions in cross-subject EEG analysis, achieving satisfactory results.
- From a geometric perspective, we accounted for differences in acquisition patterns across subjects by using quaternion rotations to mitigate distribution shifts.
- Q-RoFormer's complexity is relatively low, but it significantly reduces the number of model parameters, by 25% compared to real-valued models 🎉🎉🎉!

## Requirements:
- Python 3.11.4
- Pytorch 2.0.2
- Intel(R) Xeon(R) Gold 6226R CPU and an NVIDIA GeForce RTX 4090 GPU

## Datasets
For evaluation, the leave-one-subject-out cross-validation strategy was adopted.
- [SEED](https://bcmi.sjtu.edu.cn/~seed/seed.html) - acc 93.47% (LOSO, three classes)
- [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) - acc 81.92% (LOSO, four classes)
- We also tested it on the BCI dataset, which will be made public soon.

## Distribution Visualization
The results of Q-RoFormer are very satisfactory. After fine-tuning, the distribution shifts are significantly reduced. Detailed information can be obtained from this figure.
![Visualization](/fig5.png)

## Contact me
Quaternion Neural Networks (QNN) resource package needs to be downloaded here: [[QNN](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks)] 

After you download the qnn resource package and dataset, run the q_roformer.py file in the terminal.

If you want get some help about how to run this code, this is way: 230238579@seu.edu.cn

### Thanks
#### 🎉🎉🎉 I would like to thank Shu for his improvements to my manuscript and providing guidance on quaternion theory.
#### 🎉🎉🎉 I would like to thank Jean Louis Coatrieux for his help with quaternion theory.
#### 🎉🎉🎉 I would like to thank Regine Le Bouquin Jeannes, for his rigorous approach in formula derivation.

## Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊


