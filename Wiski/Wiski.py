#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division 
import aiwolfpy
import aiwolfpy.contentbuilder as cb

import pickle
import copy

import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras.preprocessing import sequence
from keras.preprocessing import sequence


import warnings
warnings.filterwarnings('ignore')


from predictor_5 import model as model_5
from predictor_5 import tokenizer as tokenizer_5
from predictor_15 import model as model_15
from predictor_15 import tokenizer as tokenizer_15

from first_exe import dfg, bi5, bi15


# In[2]:


myname = 'Wiski'
minimum_to_cooperate = 2
role_dict = {'VILLAGER': 0, 'SEER': 1, 'POSSESSED': 2, 'WEREWOLF': 3, 'MEDIUM': 4, 'BODYGUARD': 5}
reverse_role_dict = {v: k for k, v in role_dict.items()}
maxlen_5 = 72
maxlen_15 = 452
pred_coef = 0.7
talk_probability = 0.6


# In[3]:


def base_info_to_x2(bi):
    if len(bi['statusMap']) == 5:
        result = np.zeros((5, 4))
    else:
        result = np.zeros((15, 6))
    for k, v in bi['roleMap'].items():
        result[(int(k)-1)][role_dict[v]] = 1
    return result.reshape(-1).astype(np.float32)

def get_x1(df):
    temp_x = []
    for i, row in df[df['type'].apply(lambda r: r in ['talk', 'vote'])].iterrows():
        if row['type'] == 'vote':
            temp_x.append(' '.join([str(int(row['idx'])), 'VOTE', agent_idx_to_str(int(row['agent']))]))
        if row['type'] == 'talk':
            action = row[5].split(' ')[0]
            if action == 'COMINGOUT':
                temp_x.append(row['text'])
            elif action in ['ESTIMATE', 'DIVINED']: 
                temp_x.append(' '.join([str(int(row['agent'])), row['text']]))
            elif action == 'VOTE':
                temp_x.append(' '.join(['talk', str(int(row['agent'])), row['text']]))
    return temp_x


def predict(bi, game):
    
    for i, row in game[(game['type'] == 'divine') | (game['type'] == 'identify')].iterrows():
        if row['text'].split(' ')[-1] == 'HUMAN':
            bi['roleMap'][str(int(row['agent']))] = 'VILLAGER'
        else:
            bi['roleMap'][str(int(row['agent']))] = 'WEREWOLF'
            
    x2 = base_info_to_x2(bi)
    
    if len(bi['statusMap']) == 5:
        x1 = (sequence.pad_sequences(tokenizer_5.texts_to_sequences([get_x1(game)]), maxlen=maxlen_5)[0]).astype(np.float32)
        p = model_5.predict([x1.reshape(1, -1), x2.reshape(1, -1)]).reshape(5, 4)
    else:
        x1 = (sequence.pad_sequences(tokenizer_15.texts_to_sequences([get_x1(game)]), maxlen=maxlen_15)[0]).astype(np.float32)
        p = model_15.predict([x1.reshape(1, -1), x2.reshape(1, -1)]).reshape(15, 6)
    result = {}
    for i in range(p.shape[0]):
        result[i] = reverse_role_dict[np.argmax(p[i])]
    return result


def predict_probabilies(bi, game):
    x2 = base_info_to_x2(bi)
    if len(bi['statusMap']) == 5:
        x1 = (sequence.pad_sequences(tokenizer_5.texts_to_sequences([get_x1(agent.game)]), maxlen=maxlen_5)[0]).astype(np.float32)
        return model_5.predict([x1.reshape(1, -1), x2.reshape(1, -1)]).reshape(5, 4)
    else:
        x1 = (sequence.pad_sequences(tokenizer_15.texts_to_sequences([get_x1(agent.game)]), maxlen=maxlen_15)[0]).astype(np.float32)
        return model_15.predict([x1.reshape(1, -1), x2.reshape(1, -1)]).reshape(15, 6)


# In[4]:


def parse_vote_talk(game, day, status):
    df = (game[(game['day'] == day) & (game['type'] == 'talk') & (game['action'] == 'VOTE')]).groupby('agent').tail(1)
    votes = np.array(df.apply(lambda row: int(row['text'][-3:-1]), axis=1))
    votes_dict = {i: (votes == i).sum() for i in range(1, len(status)) if status[str(i)] == 'ALIVE'}
    return {k: v for k, v in sorted(votes_dict.items(), key=lambda item: item[1], reverse=True)}

def agent_idx_to_str(idx):
    if idx > 9:
        return 'Agent[' + str(idx) + ']'
    return 'Agent[0' + str(idx) + ']'


# ### VILLAGER

# In[10]:


class Villager(object):
    def __init__(self, base_info):
        self.talk_return = ""
        self.vote_return = base_info['agentIdx']
        self.divine_return = base_info['agentIdx']
        self.divided = [base_info['agentIdx']]
    
    
    def update(self, game, base_info, preds, agent):
        self.talk_return = ""
        self.base_info = base_info
        votes = parse_vote_talk(game, base_info['day'], base_info['statusMap'])
        
        #if voting to me
        vote_flag = True
        if len(votes.keys()) > 0:
            vote_flag = False
            if list(votes.keys())[0] == base_info['agentIdx']:
                if list(votes.values())[0] - list(votes.values())[1] <= 1:
                    self.talk_return = 'VOTE ' + agent_idx_to_str(list(votes.keys())[1])
                    self.vote_return = list(votes.keys())[1]
                else:
                    self.talk_return = ' '.join(['COMINGOUT', agent_idx_to_str(base_info['agentIdx']), 'SEER'])

            else:
                #if more the minimun voting to some player, votie to this player        
                if list(votes.values())[0] > minimum_to_cooperate:
                    self.talk_return = 'VOTE ' + agent_idx_to_str(list(votes.keys())[0])
                    self.vote_return = list(votes.keys())[0]
                else:
                    vote_flag = True
                    
        if vote_flag:
            #vote by voting model with talk probability
            v = agent.voting_model('WEREWOLF')
            self.vote_return = v
            if base_info['myRole'] == 'SEER' and len(game[game['type'] == 'divine']) > 0:
                self.talk_return = game[game['type'] == 'divine'].iloc[-1]['text']
            elif base_info['myRole'] == 'MEDIUM' and len(game[game['type'] == 'identify']) > 0:
                self.talk_return = game[game['type'] == 'identify'].iloc[-1]['text']
            elif np.random.random() <= talk_probability:
                self.talk_return = 'VOTE ' + agent_idx_to_str(v)
        
    def dayStart(self):
        self.talk_return = ""
        return None

    def talk(self):
        if self.talk_return != "":
            return self.talk_return
        return cb.skip()

    def vote(self):
        return self.vote_return
    
    
    def divine(self, agent):
        for k, v in dict(agent.game[agent.game['action'] == 'DIVINED']['idx'].astype(int).value_counts()).items():
            k = int(k)
            if v not in self.divided:
                self.divided.append(v)
                return v
        v = agent.voting_model('WEREWOLF')
        if v not in self.divided:
            self.divided.append(v)
            return v
        for v in range(1, agent.game_setting['playerNum']+1):
            if v not in self.divided:
                self.divided.append(v)
                return v

    def guard(self):
        return agent.voting_model('SEER')


# ### WEREWOLF

# In[14]:


class Werewolf(object):
    def __init__(self, base_info):
        self.talk_return = ""
        self.vote_return = base_info['agentIdx']
        self.attack_return = base_info['agentIdx']
        self.base_info = base_info
    
    
    def update(self, game, base_info, preds, agent):
        self.talk_return = ""
        self.base_info = base_info
        votes = parse_vote_talk(game, base_info['day'], base_info['statusMap'])
        
        #if voting to me
        vote_flag = True
        if len(votes.keys()) > 0:
            vote_flag = False
            if list(votes.keys())[0] == base_info['agentIdx']:
                if list(votes.values())[0] - list(votes.values())[1] <= 1:
                    self.talk_return = 'VOTE ' + agent_idx_to_str(list(votes.keys())[1])
                    self.vote_return = list(votes.keys())[1]
                else:
                    self.talk_return = ' '.join(['COMINGOUT', agent_idx_to_str(base_info['agentIdx']), 'SEER'])

            else:
                #if more the minimun voting to some player, votie to this player        
                if list(votes.values())[0] > minimum_to_cooperate:
                    self.talk_return = 'VOTE ' + agent_idx_to_str(list(votes.keys())[0])
                    self.vote_return = list(votes.keys())[0]
                else:
                    vote_flag = True
                    
        if vote_flag:
            #vote by voting model with talk probability
            v = agent.voting_model('SEER')
            if str(v) in agent.base_info['roleMap'].keys():
                if agent.base_info['roleMap'][str(v)] == 'WEREWOLF':
                    for i in range(1, agent.game_setting['playerNum']+1):
                        if str(i) not in agent.base_info['roleMap'].keys():
                            self.vote_return = i
                            break
                            
                
    def dayStart(self):
        self.talk_return = ""
        return None

    def talk(self):
        if self.talk_return != "":
            return self.talk_return
        return cb.skip()

    def vote(self):
        return self.vote_return
    
    def attack(self):
        return self.attack_return

    def whisper(self, agent):
        to_attack = agent.voting_model('SEER')
        if str(to_attack) in agent.base_info['roleMap'].keys():
            if agent.base_info['roleMap'][str(to_attack)] == 'WEREWOLF':
                for i in range(1, agent.game_setting['playerNum']+1):
                    if str(i) not in agent.base_info['roleMap'].keys():
                        to_attack = i
                        break
        self.attack_return = to_attack
        return 'VOTE ' + agent_idx_to_str(to_attack)
        return cb.skip()


# ### AGENT

# In[15]:


class Agent(object):
    
    def __init__(self, agent_name):
        self.myname = agent_name
        self.role = None   
        self.is_first_game = True

        predict(bi5, dfg);
        predict(bi15, dfg);
        
    def getName(self):
        return self.myname
    
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting
        self.game = diff_data
        self.game['action'] = self.game.apply(lambda row: row['text'].split(' ')[0], axis=1)
        if base_info['myRole'] in ['VILLAGER', 'SEER', 'BODYGUARD', 'MEDIUM']:
            self.role = Villager(base_info)
        if base_info['myRole'] in ['WEREWOLF', 'POSSESSED']:
            self.role = Werewolf(base_info)
        if self.is_first_game == True:
            self.win_rates = {str(k): 0 for k in range(1, game_setting['playerNum']+1)}
            self.win_rates[base_info['agentIdx']] = -1
            self.is_first_game = False
        
        
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        if len(diff_data) > 0:
            diff_data['action'] = diff_data.apply(lambda row: row['text'].split(' ')[0],axis=1)
        self.game = self.game.append(diff_data)
        self.role.update(self.game, self.base_info, 1, self)

    def dayStart(self):     
        return self.role.dayStart()
    
    def talk(self):   
        return self.role.talk()
    
    def whisper(self):
        return self.role.whisper(self)
        
    def vote(self):   
        return self.role.vote()
    
    def attack(self):
        return self.role.attack()
    
    def divine(self):
        return self.role.divine(self)
    
    def guard(self): 
        return self.role.guard()
    
    def finish(self):
        for k, v in self.base_info['statusMap'].items():
            if v == 'ALIVE' and int(k) != self.base_info['agentIdx']:
                self.win_rates[k] += 1
        return None
    
    def voting_model(self, role='WEREWOLF'):
        pred = predict_probabilies(self.base_info, self.game).T[role_dict[role]]
        pred = pred / pred.sum()
        win_sum = sum(self.win_rates.values())
        wr = np.array(list(self.win_rates.values())) / win_sum
        voting_scores = (pred_coef*pred) + ((1-pred_coef)*win_sum)
        voting_list = np.argsort(-voting_scores) + 1
        for i in voting_list:
            if i != self.base_info['agentIdx'] and self.base_info['statusMap'][str(i)] == 'ALIVE':
                return i
        return self.base_info['agentIdx']


# In[16]:


agent = Agent(myname)

aiwolfpy.connect_parse(agent)


# In[311]:


agent.game['text'].value_counts()


# In[372]:


agent.base_info


# In[353]:


agent.base_info['roleMap']['5']


# In[355]:


'4' 


# In[ ]:




