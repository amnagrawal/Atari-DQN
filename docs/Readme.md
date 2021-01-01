**Paper Title:** Human-level control through deep reinforcement learning 

**Authors:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg and Demis Hassabis 

**Problem:** 

The theory of reinforcement learning deals with how artificial agents make intelligent decisions or actions in an environment in order to maximize their reward. But, to use this successfully in situations approaching real-world complexity, the agents have a difficult task to derive efficient representations of the environment. 

The goal of this project is to provide an artificial agent a gamified environment, with a certain set of rules for receiving rewards for taking actions that could maximize them. Eventually, demonstrate how this agent can learn to do this task in real time. 

**Approach**:  

The recent advances in training deep neural networks to develop an artificial agent capable of taking intelligent decisions, by learning successful policies using end-to-end reinforcement learning.  

The deep Q-network agent is provided with the environments of classic Atari 2600 games. The agent receives environment information in the form of pixels through the environment image at any particular state, and the game score at that state. The agent is challenged to maximize this score, through a combination of techniques like deep convolution network architecture, experience replay and deep Q-network. 

Through this project I aim to demonstrate that using only the information mentioned above, the deep Q-network is able to achieve or in some case surpass the level of a professional human games tester. 

**Data:**  

The networks are trained to run on the environments available in the open-AI gym library. In this project I have trained the model on following 2 environments: 

- Space Invaders 
- Breakout 

**Running the project:**  

Install the following python packages before running the project 

- Open-AI gym 
- Keras 
- Keras-RL 
- Tensorflow 
- Pickle 

Run the main.py file with following arguments: 

- -- mode: train or test (for training or testing respectively, default=train) 
- -- weights: ‘weights\_file’ (for using the weights trained before, default=None) 
- -- env-name: ‘name\_of\_the\_gym\_environment’ (default=BreakoutDeterministic-v4) 

**Implementation details:** 

- Learn values Q(s, a) from pixels 
- Input state is stack of raw pixels from last 4 frames 
- Output is 4-18 dimensional vector, depending on the action-space of the game being played 
- Reward is change in score for that step 
- The raw Atari frames are 210 x 160 pixel images, and it is computationally expensive to train the model on these images 
- Thus, images are preprocessed to reduce dimensionality 
- To preprocess the images, the paper suggests to follow these steps : 
- Encode a single frame by taking max value for each pixel color over frame being encoded and previous frame. It is done to remove flickering when some objects appear in even frames and some in odd frames. 
- Convert the images to grayscale and rescale the image to 84 x 84 
- Stack 4 recent preprocessed frames as an input to the network 

**Network Architecture:**  

<center>Input (84 x 84 x 4) </center>
<center>↓ </center>

<center>Conv2D (32, (8, 8), Activation=relu) </center>

<center>↓ </center>

<center>Conv2D (64, (4, 4), Activation=relu) </center>

<center>↓ </center>

<center>Conv2D (64, (3, 3), Activation=relu) </center>

<center>↓ </center>

<center>Flatten() </center>

<center>↓ </center>

<center>Dense (512, Activation=relu) </center>

<center>↓ </center>

<center>Dense (1, Activation=linear) </center>

<center>↓ </center>
<center>Output (Policy=epsilon-greedy policy, Optimizer=RMSprop, metrics=mae) </center>





- Above is the network as suggested in the paper 
- Epsilon-greedy policy is used to train the Deep Q-Network 
- The epsilon value is linearly annealed from 0.9 to 0.1. This is done, so that the model initially trains more by taking random actions and exploring the state space, and later continue with what it knows 
- Atari environments used to train the DQN are: BreakoutDeterministic-v4, SpaceInvadersDeterministic-v4 
- The above environments can exploit the capability of FrameSkipping, which helps to speed up the training process 

**Results:** 

- Breakout game was trained for 800k time steps 
- Space Invader game was trained for 4M time steps 
- Some of these training results are saved as video files in the following directory:  <https://drive.google.com/open?id=1CfSexyPkhrnclTXAUXYN4RpnPPxQP1PB>  

**Inferences:** 

- It can be observed that the training begins with the model knowing nothing about the environment 
- By the end of the training the model is able to play the game much decently. 

**References:** 

- <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>  
- [https://deepmind.com/research/publications/human-level-control-through-deep-reinforce ment-learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) 
- <https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning> 
- <https://www.davidsilver.uk/teaching/> 
- [https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPe bj2MfCFzFObQ](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 
