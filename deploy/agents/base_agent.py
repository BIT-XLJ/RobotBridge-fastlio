import time
import numpy as np
import torch
from loguru import logger

class BaseAgent:
    def __init__(self, config, env):
        self.config = config
        self.time = 0
        self.print_cnt = 0
        self.env = env
        self.device = self.config.get('device', 'cpu')
        
        self.load_policy()

    def load_policy(self):
        ckpt_path = self.config.checkpoint
        self.policy = torch.jit.load(ckpt_path).to(self.device)
        logger.info(f'Loading Checkpoint from {ckpt_path}')
    
    def run(self):
        self.time = time.time()
        obs = self.env.reset()
        while True:
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.policy(obs).detach().cpu().numpy()
            start=time.time()
            obs = self.env.step(action)
            # if (self.print_cnt < 5):
                # logger.info(f'Step {self.print_cnt}')
                # logger.info(f'The policy layer runs at a frequency of {1/(time.time() - self.time)}HZ')
                # logger.info(f'One step takes {time.time()-self.time}')
                # self.print_cnt+=1
            time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
            if time_until_next_step>0:
                time.sleep(time_until_next_step)
            self.time=time.time()