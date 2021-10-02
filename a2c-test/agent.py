import torch
import numpy as np
from tqdm import trange
from torch.distributions import Categorical


class ActorCriticAgent:
    ''' env should be gym environment '''
    def __init__(self, name, env, model, optimizer, num_episode, max_step, gamma=0.99, device='cpu'):
        self.name = name
        self.env = env
        self.model = model
        self.optimizer = optimizer

        self.num_episode = num_episode
        self.max_step = max_step
        self.gamma = gamma
        self.device = device

        self.model.to(self.device)

    def get_action(self, policy_dist):
        ''' should return action '''
        pass

    def step(self, action):
        ''' step through env, return new state, reward, done '''
        pass

    def reset(self):
        ''' should return new state after reset '''
        pass

    def train(self):
        self.model.train()
        entropy_term = 0
        all_rewards = []
        all_lens = []

        for episode in (tq := trange(self.num_episode)):
            state = self.reset()
            rewards = []
            log_action_probs = []
            values = []

            for step in range(self.max_step):
                value_tensor, policy_dist = self.model(state)
                action = self.get_action(policy_dist)

                state, reward, done = self.step(action)
                
                rewards.append(reward)
                values.append(value_tensor)

                log_prob = torch.log(policy_dist[0, action])
                log_action_probs.append(log_prob)

                policy_dist_np = policy_dist.cpu().detach().numpy()
                entropy = np.sum(np.mean(policy_dist_np) * np.log(policy_dist_np))
                entropy_term += entropy

                if done or step == self.max_step - 1:
                    Q_value = value_tensor.cpu().detach().item()
                    all_rewards.append(sum(rewards))
                    all_lens.append(step)
                    if episode % 10 == 0:
                        tq.set_description(f'Episode: {episode}, reward: {sum(rewards)}, total length: {step}')

                    break

            Q_values = np.zeros_like(values, dtype='float32')
            for t in reversed(range(len(rewards))):
                Q_value = rewards[t] + self.gamma * Q_value
                Q_values[t] = Q_value

            values = torch.tensor(values, dtype=torch.float32, device=self.device)
            Q_values = torch.tensor(Q_values, dtype=torch.float32, device=self.device)
            log_action_probs = torch.stack(log_action_probs)

            advantage = Q_values - values
            actor_loss = (-log_action_probs * advantage).mean()
            # MSE
            critic_loss = 0.5 * advantage.pow(2).mean()
            
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()
        
        torch.save(self.model.state_dict(), f'{self.name}.pth')

    def play(self, render=True):
        self.model.eval()

        with torch.no_grad():
            rewards = 0
            state = self.reset()
            for step in (t := trange(self.max_step)):
                if render:
                    self.env.render()

                _, policy_dist = self.model(state)
                action = self.get_action(policy_dist)

                state, reward, done = self.step(action)
                rewards += reward

                t.set_description(f'reward: {rewards}, step: {step}')

                if done:
                    break


class CartPoleAgent(ActorCriticAgent):
    def get_action(self, policy_dist):
        probs = Categorical(policy_dist)
        action = probs.sample().cpu().detach().item()

        return action

    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)

        return new_state, reward, done

    def reset(self):
        state = self.env.reset()

        return state
