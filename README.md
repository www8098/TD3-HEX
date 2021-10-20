# TD3-HEX
This project applies TD3 model to control a hexpoad robot wlking to a fixed target position. The entire project includes the following files:

## memory.py
reference: https://github.com/ghliu/pytorch-ddpg/blob/master/memory.py  
Self defined RingBuffer and Obervation Memmory. Store the observations (state action , next_state, reward, done) in it at the end of every step.
