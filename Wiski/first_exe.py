import numpy as np
import pandas as pd

g = np.array(['talk 5 VOTE Agent[01]',
        'talk 2 VOTE Agent[01]',
        'COMINGOUT Agent[03] SEER',
        '1 ESTIMATE Agent[03] SEER',
        'talk 5 VOTE Agent[02]',
        'talk 3 VOTE Agent[01]',
        '1 ESTIMATE Agent[03] SEER',
        'talk 5 VOTE Agent[03]',
        'talk 1 VOTE Agent[05]',
        'talk 3 VOTE Agent[05]',
        '1 ESTIMATE Agent[03] SEER',
        'talk 2 VOTE Agent[05]',
        'talk 3 VOTE Agent[05]',
        'talk 3 VOTE Agent[05]',
        '1 ESTIMATE Agent[02] WEREWOLF',
        '1 ESTIMATE Agent[04] WEREWOLF',
        'talk 3 VOTE Agent[05]',
        '1 VOTE Agent[05]',
        '2 VOTE Agent[05]',
        '3 VOTE Agent[05]',
        '4 VOTE Agent[01]',
        '5 VOTE Agent[03]',
        'talk 2 VOTE Agent[03]',
        '3 ESTIMATE Agent[02] WEREWOLF',
        'talk 3 VOTE Agent[02]',
        'talk 3 VOTE Agent[02]',
        '3 ESTIMATE Agent[02] WEREWOLF',
        '2 VOTE Agent[03]',
        '3 VOTE Agent[02]',
        '4 VOTE Agent[03]'])

lst = [['day', 'type', 'idx', 'turn', 'agent', 'text', 'action'],
 ['0.0',
  'initialize',
  '5.0',
  '0.0',
  '5.0',
  'COMINGOUT Agent[05] VILLAGER',
  'COMINGOUT'],
 ['1.0', 'talk', '0.0', '0.0', '3.0', 'Skip', 'Skip'],
 ['1.0', 'talk', '1.0', '0.0', '4.0', 'COMINGOUT Agent[04] SEER', 'COMINGOUT'],
 ['1.0', 'talk', '2.0', '0.0', '2.0', 'COMINGOUT Agent[02] SEER', 'COMINGOUT'],
 ['1.0', 'talk', '3.0', '0.0', '1.0', 'COMINGOUT Agent[01] SEER', 'COMINGOUT'],
 ['1.0', 'talk', '4.0', '0.0', '5.0', 'Skip', 'Skip'],
 ['1.0', 'talk', '5.0', '1.0', '1.0', 'DIVINED Agent[02] WEREWOLF', 'DIVINED'],
 ['1.0', 'talk', '6.0', '1.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '7.0', '1.0', '2.0', 'DIVINED Agent[01] HUMAN', 'DIVINED'],
 ['1.0', 'talk', '8.0', '1.0', '4.0', 'DIVINED Agent[01] WEREWOLF', 'DIVINED'],
 ['1.0', 'talk', '9.0', '1.0', '5.0', 'Skip', 'Skip'],
 ['1.0', 'talk', '10.0', '2.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '11.0', '2.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '12.0', '2.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '13.0', '2.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '14.0', '2.0', '2.0', 'VOTE Agent[04]', 'VOTE'],
 ['1.0', 'talk', '15.0', '3.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '16.0', '3.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '17.0', '3.0', '2.0', 'Skip', 'Skip'],
 ['1.0', 'talk', '18.0', '3.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '19.0', '3.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '20.0', '4.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '21.0', '4.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '22.0', '4.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '23.0', '4.0', '2.0', 'Skip', 'Skip'],
 ['1.0', 'talk', '24.0', '4.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '25.0', '5.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '26.0', '5.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '27.0', '5.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '28.0', '5.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '29.0', '5.0', '2.0', 'Over', 'Over'],
 ['1.0', 'talk', '30.0', '6.0', '2.0', 'Over', 'Over'],
 ['1.0', 'talk', '31.0', '6.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '32.0', '6.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '33.0', '6.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '34.0', '6.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '35.0', '7.0', '2.0', 'Over', 'Over'],
 ['1.0', 'talk', '36.0', '7.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '37.0', '7.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '38.0', '7.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '39.0', '7.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '40.0', '8.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '41.0', '8.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '42.0', '8.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '43.0', '8.0', '2.0', 'Over', 'Over'],
 ['1.0', 'talk', '44.0', '8.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '45.0', '9.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '46.0', '9.0', '2.0', 'Over', 'Over'],
 ['1.0', 'talk', '47.0', '9.0', '4.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'talk', '48.0', '9.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '49.0', '9.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '50.0', '10.0', '1.0', 'Over', 'Over'],
 ['1.0', 'talk', '51.0', '10.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '52.0', '10.0', '2.0', 'Over', 'Over'],
 ['1.0', 'talk', '53.0', '10.0', '4.0', 'Over', 'Over'],
 ['1.0', 'talk', '54.0', '10.0', '3.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'talk', '55.0', '11.0', '3.0', 'Over', 'Over'],
 ['1.0', 'talk', '56.0', '11.0', '5.0', 'Over', 'Over'],
 ['1.0', 'talk', '57.0', '11.0', '4.0', 'Over', 'Over'],
 ['1.0', 'talk', '58.0', '11.0', '1.0', 'Over', 'Over'],
 ['1.0', 'talk', '59.0', '11.0', '2.0', 'Over', 'Over'],
 ['1.0', 'vote', '1.0', '-1.0', '2.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'vote', '2.0', '-1.0', '4.0', 'VOTE Agent[04]', 'VOTE'],
 ['1.0', 'vote', '3.0', '-1.0', '2.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'vote', '4.0', '-1.0', '1.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'vote', '5.0', '-1.0', '4.0', 'VOTE Agent[04]', 'VOTE'],
 ['1.0', 'vote', '1.0', '0.0', '2.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'vote', '2.0', '0.0', '4.0', 'VOTE Agent[04]', 'VOTE'],
 ['1.0', 'vote', '3.0', '0.0', '2.0', 'VOTE Agent[02]', 'VOTE'],
 ['1.0', 'vote', '4.0', '0.0', '1.0', 'VOTE Agent[01]', 'VOTE'],
 ['1.0', 'vote', '5.0', '0.0', '4.0', 'VOTE Agent[04]', 'VOTE'],
 ['1.0', 'execute', '0.0', '0.0', '4.0', 'Over', 'Over'],
 ['2.0', 'dead', '0.0', '0.0', '3.0', 'Over', 'Over'],
 ['2.0', 'talk', '0.0', '0.0', '5.0', 'Skip', 'Skip'],
 ['2.0', 'talk', '1.0', '0.0', '2.0', 'DIVINED Agent[05] HUMAN', 'DIVINED'],
 ['2.0', 'talk', '2.0', '0.0', '1.0', 'DIVINED Agent[05] WEREWOLF', 'DIVINED'],
 ['2.0',
  'talk',
  '3.0',
  '1.0',
  '1.0',
  'COMINGOUT Agent[01] WEREWOLF',
  'COMINGOUT'],
 ['2.0',
  'talk',
  '4.0',
  '1.0',
  '2.0',
  'COMINGOUT Agent[02] WEREWOLF',
  'COMINGOUT'],
 ['2.0', 'talk', '5.0', '1.0', '5.0', 'Skip', 'Skip'],
 ['2.0', 'talk', '6.0', '2.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '7.0', '2.0', '2.0', 'VOTE Agent[01]', 'VOTE'],
 ['2.0', 'talk', '8.0', '2.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '9.0', '3.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '10.0', '3.0', '2.0', 'Skip', 'Skip'],
 ['2.0', 'talk', '11.0', '3.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '12.0', '4.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '13.0', '4.0', '2.0', 'Skip', 'Skip'],
 ['2.0', 'talk', '14.0', '4.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '15.0', '5.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '16.0', '5.0', '2.0', 'Over', 'Over'],
 ['2.0', 'talk', '17.0', '5.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '18.0', '6.0', '2.0', 'Over', 'Over'],
 ['2.0', 'talk', '19.0', '6.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '20.0', '6.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '21.0', '7.0', '2.0', 'Over', 'Over'],
 ['2.0', 'talk', '22.0', '7.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '23.0', '7.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '24.0', '8.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '25.0', '8.0', '2.0', 'Over', 'Over'],
 ['2.0', 'talk', '26.0', '8.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '27.0', '9.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '28.0', '9.0', '2.0', 'Over', 'Over'],
 ['2.0', 'talk', '29.0', '9.0', '1.0', 'VOTE Agent[02]', 'VOTE'],
 ['2.0', 'talk', '30.0', '10.0', '1.0', 'Over', 'Over'],
 ['2.0', 'talk', '31.0', '10.0', '5.0', 'Over', 'Over'],
 ['2.0', 'talk', '32.0', '10.0', '2.0', 'Over', 'Over'],
 ['3.0',
  'finish',
  '1.0',
  '0.0',
  '1.0',
  'COMINGOUT Agent[01] WEREWOLF',
  'COMINGOUT'],
 ['3.0',
  'finish',
  '2.0',
  '0.0',
  '2.0',
  'COMINGOUT Agent[02] POSSESSED',
  'COMINGOUT'],
 ['3.0',
  'finish',
  '3.0',
  '0.0',
  '3.0',
  'COMINGOUT Agent[03] VILLAGER',
  'COMINGOUT'],
 ['3.0',
  'finish',
  '4.0',
  '0.0',
  '4.0',
  'COMINGOUT Agent[04] SEER',
  'COMINGOUT'],
 ['3.0',
  'finish',
  '5.0',
  '0.0',
  '5.0',
  'COMINGOUT Agent[05] VILLAGER',
  'COMINGOUT']]
 
dfg = pd.DataFrame(lst)
dfg.columns = dfg.iloc[0]
dfg = dfg[1:]
dfg['day'] = dfg['day'].astype(np.float64)
dfg['idx'] = dfg['idx'].astype(np.float64)
dfg['turn'] = dfg['turn'].astype(np.float64)
dfg['agent'] = dfg['agent'].astype(np.float64)

bi5 = {'agentIdx': 5,
 'myRole': 'VILLAGER',
 'roleMap': {'5': 'VILLAGER'},
 'day': 3,
 'remainTalkMap': {'1': 10},
 'remainWhisperMap': {},
 'statusMap': {'1': 'ALIVE',
  '2': 'DEAD',
  '3': 'DEAD',
  '4': 'DEAD',
  '5': 'DEAD'}}
  
bi15 = {'agentIdx': 15,
 'myRole': 'VILLAGER',
 'roleMap': {'15': 'VILLAGER'},
 'day': 0,
 'remainTalkMap': {'1': 10,
  '2': 10,
  '3': 10,
  '4': 10,
  '5': 10,
  '6': 10,
  '7': 10,
  '8': 10,
  '9': 10,
  '10': 10,
  '11': 10,
  '12': 10,
  '13': 10,
  '14': 10,
  '15': 10},
 'remainWhisperMap': {},
 'statusMap': {'1': 'ALIVE',
  '2': 'ALIVE',
  '3': 'ALIVE',
  '4': 'ALIVE',
  '5': 'ALIVE',
  '6': 'ALIVE',
  '7': 'ALIVE',
  '8': 'ALIVE',
  '9': 'ALIVE',
  '10': 'ALIVE',
  '11': 'ALIVE',
  '12': 'ALIVE',
  '13': 'ALIVE',
  '14': 'ALIVE',
  '15': 'ALIVE'}}