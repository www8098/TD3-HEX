# TD3-HEX
This project applies TD3 model to control a hexpoad robot wlking to a fixed target position. The entire project includes the following files:

## memory.py
reference: https://github.com/ghliu/pytorch-ddpg/blob/master/memory.py  
Self defined RingBuffer and Obervation Memmory. Store the observations (state action , next_state, reward, done) in it at the end of every step.

## parameters.py  
This part use paras to set all the projects' configurable parameters, including training hyper parameters (learning rate, exploration rate, decay rate, number of neurons, etc) and environment configurations (gym environment name, total steps, evaluation steps, etc.)

## model.py  
It sets up the 3-layer actor and critic with fixed number of neurons, that is configured in parameters.py. The actor uses tanh as active function while the critic applies relu. These classes use pytorch's default xavier weight initialize method. 

## td3.py  
This is the code for TD3 framework. I build the framework by modifying DDPG code (reference \emph{https://github.com/ghliu/pytorch-ddpg}). According to the paper of Twin Delay Deterministic Policy Gradient \cite{fujimoto2018addressing}, I wrote my own codes for policy noise, action noise, clipped double Q network , etc.

## evaluatoe.py  
This module comes from \emph{https://github.com/ghliu/pytorch-ddpg} which is for evaluating the model's performance during training and test process. I added gaits coupling code to fit my algorithm design.

## trainer.py  
I wrote this module with reference to the training code of some open source projects (reference \emph{https://github.com/ghliu/pytorch-ddpg} and \emph{https://github.com/mit-han-lab/haq}). This module consists of three main functions: train, test and bchavior clone. When train or evaluate the agent, it will call one of the three functions according to the argument "mode" in \emph{parameters.py}. Besides I added the code for gait coupling for comparison experiments.

## random_process.py
This is the code for the stochastic process, where the Ornstein Uhlenbeck Process is used to generate the exploration noise in the footballs process. It refers to \emph{https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py}.

## nomarlized_env.py
This module includes some useful gym wrapper like action normalize wrapper, reward clip wrapper and observation normalize wrapper. I wrote this module following the official gym gihub instructions (https://github.com/openai/gym/blob/master/gym/core.py).
