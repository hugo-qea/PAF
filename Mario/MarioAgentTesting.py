from mario_dqn import Mario, env , save_dir



from pathlib import Path

import datetime, os, copy

import torch

from torch import nn

import torchvision.models as models




# Chargement de l'environnement du jeu, de l'émulateur

# In[3]:



mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

checkpoint = torch.load("./mario_net_0.chkpt")
ordDict = checkpoint['model']

mario.net.load_state_dict(ordDict) 

#mario.net.eval()

mario.exploration_rate = 0

print("######### Trained V #################")

while True :
	state = env.reset()
	# Play the game!
	while True:
		# Run agent on the state
		action = mario.act(state)
		# Agent performs action
		next_state, reward, done, info = env.step(action)
		state = next_state
		# Check if end of game
		env.render()
		if done or info["flag_get"]:
			break



