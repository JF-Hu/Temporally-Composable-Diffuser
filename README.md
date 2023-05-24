# Instructed Diffuser with Temporal Condition Guidance for Offline Reinforcement Learning
<div align="center">

![Architecture of TCD](framework.png)



</div>

Recent works have shown the potential of diffusion models in computer vision and natural language processing. Apart from the classical supervised learning fields, diffusion models have also shown strong competitiveness in reinforcement learning (RL) by formulating decision-making as sequential generation. However, incorporating temporal information of sequential data and utilizing it to guide diffusion models to perform better generation is still an open challenge. In this paper, we take one step forward to investigate controllable generation with temporal conditions that are refined from temporal information. We observe the importance of temporal conditions in sequential generation in sufficient explorative scenarios and provide a comprehensive discussion and comparison of different temporal conditions. Based on the observations, we propose an effective temporally-conditional diffusion model coined Temporally-Composable Diffuser (TCD), which extracts temporal information from interaction sequences and explicitly guides generation with temporal conditions. Specifically, we separate the sequences into three parts according to time expansion and identify historical, immediate, and prospective conditions accordingly. Each condition preserves non-overlapping temporal information of sequences, enabling more controllable generation when we jointly use them to guide the diffuser. Finally, we conduct extensive experiments and analysis to reveal the favorable applicability of TCD in offline RL tasks, where our method reaches or matches the best performance compared with prior SOTA baselines.

We provide training code, evaluation code, and environmental installation instruction for Temporally-Composable Diffuser (TCD).

# Installation
```
1.Install D4RL dataset, including MuJoCo, Gym, dm-control, and mjrl. 
2.Install the packages listed in the requirement.txt
```

# Training and Evaluation
To start training the model, run the following command

```
python3 model_train.py
```

To evaluate the trained model, run the following command
```
python3 model_eval.py
```

To train diffuser with various temporal conditions, please check the mapping relationship bellow and modify the hyperparameter configuration in locomotion_config.py

| Config.return_type | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| method|DD|DD with hard reward|RR-TCD|RQR-TCD|TFD|DQD with hard reward|DQD|SRD|TCD|

If the hyperparameter configuration is inappropriate, please check the error information and amend the configuration.



# Acknowledgements

The codebase is derived from [diffuser](https://github.com/jannerm/diffuser/) and [decision diffuser](https://github.com/anuragajay/decision-diffuser).