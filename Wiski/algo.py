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
		
#in initialize function		
self.game = diff_data
self.game['action'] = self.game.apply(lambda row: row['text'].split(' ')[0], axis=1)
		
#in update function
if len(diff_data) > 0:
	diff_data['action'] = diff_data.apply(lambda row: row['text'].split(' ')[0],axis=1)
	
	
	
	
	
	
#exe first
import numpy as np
import pandas as pd
from predictor_5 import model as model_5
from predictor_5 import tokenizer as tokenizer_5
from predictor_15 import model as model_15
from predictor_15 import tokenizer as tokenizer_15

from first_exe import dfg, bi5, bi15

predict(bi5, dfg);
predict(bi15, dfg);