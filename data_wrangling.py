import pandas as pd
import os
import feather
import numpy as np
import warnings

warnings.filterwarnings("ignore")
cwd = os.getcwd()

data_folder = cwd + '//project_data'
data = pd.DataFrame()
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if '.csv' in file:
            data = data.append(pd.read_csv(data_folder + '//' + file,low_memory=False),ignore_index=True)


#players = [2544,201142,201939,201935]
#players being considered
#LeBron James(2544)
#Kevin Durant(201142)
#Steph Curry(201939)
#James Harden(201935)



data['SCOREMARGIN'] = data.groupby('GAME_ID')['SCOREMARGIN'].fillna(method='ffill')
data['SCORE'] = data.groupby('GAME_ID')['SCORE'].fillna(method='ffill')

data = data.reset_index(drop=True)

data.to_feather('full_data.feather')


#data = feather.read_dataframe('full_data.feather')

players = [2544,201142,201939,201935]
player_data = data[(data['PLAYER1_ID'].isin(players)) | (data['PLAYER2_ID'].isin(players)) | (data['PLAYER3_ID'].isin(players))]
player_data['SCOREMARGIN'] = player_data['SCOREMARGIN'].apply(lambda x: float(str(x).replace('TIE','0')))
#Seperating clutch situations from non clutch situations
player_data['CRUNCH'] = 0
for i, r in player_data.iterrows():
    if r['PERIOD'] < 4:
        continue
    if pd.isnull(r['SCOREMARGIN']):
        continue
    elif abs(r['SCOREMARGIN']) < 10 and r['PERIOD'] >= 4:
        player_data.at[i,'CRUNCH'] = 1

player_data = player_data.reset_index(drop=True)
player_data.to_feather('player_data.feather')

player_data = feather.read_dataframe('player_data.feather')

game_id_details = player_data[['GAME_ID','PERIOD']].groupby(by=['GAME_ID']).agg(['max']).reset_index().T.reset_index(drop=True).T.rename(columns={0:'GAME_ID',1:'TOTAL_PERIODS'})
games_with_crunch_time = player_data[player_data['CRUNCH']==1]['GAME_ID'].unique()
games_without_crunch_time = list(set(player_data[player_data['CRUNCH']==0]['GAME_ID'].unique()) - set(games_with_crunch_time))



player_data['VISITORDESCRIPTION'] = player_data['VISITORDESCRIPTION'].fillna('')
player_data['HOMEDESCRIPTION'] = player_data['HOMEDESCRIPTION'].fillna('')
player_data['DESCRIPTION'] = player_data['VISITORDESCRIPTION'] + player_data['HOMEDESCRIPTION']

player_id_dict = {2544:'LeBron James',201142:'Kevin Durant',201939:'Steph Curry',201935:'James Harden'}

#filtering only the metrics/plays that we are considering
#1. shot made, shots missed
#2. Assists
#3. Rebounds(Offensive/defensive)
#4. Freethrows made, missed
#5. Fouls
#6. Turnovers

def getstatsplayers(plid):
    tempplayer_data = player_data[(player_data['PLAYER1_ID']==plid) | (player_data['PLAYER2_ID']==plid) | (player_data['PLAYER3_ID']==plid)]
    stats = {'2P_ATTEMPTS':0,'2P_MADE':0,'3P_ATTEMPTS':0,'3P_MADE':0,'FTA':0,'FTM':0,'REBOUNDS':0,'FOULS':0,'ASSISTS':0,'TURNOVERS':0,'BLOCKS':0}
    player_dict = {k:{0:stats.copy(),1:stats.copy()} for k in tempplayer_data['GAME_ID'].unique()}

    for i, r in tempplayer_data.iterrows():
        if r['SHOT_PLAYER_ID'] == plid:
            if '3PT' in r['DESCRIPTION']:
                player_dict[r['GAME_ID']][r['CRUNCH']]['3P_ATTEMPTS'] += 1
                if r['SHOT_MADE']:
                    player_dict[r['GAME_ID']][r['CRUNCH']]['3P_MADE'] += 1
            else:
                player_dict[r['GAME_ID']][r['CRUNCH']]['2P_ATTEMPTS'] += 1
                if r['SHOT_MADE']:
                    player_dict[r['GAME_ID']][r['CRUNCH']]['2P_MADE'] += 1
        if r['FREE_THROW_PLAYER_ID'] == plid:
            player_dict[r['GAME_ID']][r['CRUNCH']]['FTA'] += 1
            if r['FREE_THROW_MADE']:
                player_dict[r['GAME_ID']][r['CRUNCH']]['FTM'] += 1
        if r['ASSIST_PLAYER_ID'] == plid:
            player_dict[r['GAME_ID']][r['CRUNCH']]['ASSISTS'] += 1
        if r['REBOUND_PLAYER_ID'] == plid:
            player_dict[r['GAME_ID']][r['CRUNCH']]['REBOUNDS'] += 1
        if r['TURNOVER_PLAYER_ID'] == plid:
            player_dict[r['GAME_ID']][r['CRUNCH']]['TURNOVERS'] += 1
        if r['BLOCK_PLAYER_ID'] == plid:
            player_dict[r['GAME_ID']][r['CRUNCH']]['BLOCKS'] += 1
        if r['FOULED_BY_PLAYER_ID'] == plid:
            player_dict[r['GAME_ID']][r['CRUNCH']]['FOULS'] += 1
    return(player_dict)


detailed_data = pd.DataFrame()
for plid in [2544,201142,201939,201935]:
    temp = getstatsplayers(plid) #dict with each game as key and v={0:stats,1:stats}
    for game,allstats in temp.items():
        for crunch,stats in allstats.items():
            if stats == {'2P_ATTEMPTS':0,'2P_MADE':0,'3P_ATTEMPTS':0,'3P_MADE':0,'FTA':0,'FTM':0,'REBOUNDS':0,'FOULS':0,'ASSISTS':0,'TURNOVERS':0,'BLOCKS':0}:
                continue
            stats = {k:[v] for k,v in stats.items()}
            df_temp = pd.DataFrame(stats)
            df_temp['CRUNCH'] = crunch
            df_temp['PLAYER'] = player_id_dict[plid]
            df_temp['GAME_ID'] = game
            detailed_data = detailed_data.append(df_temp,ignore_index=True)
        
        
            
detailed_data['2P_PERCENT'] = detailed_data.apply(lambda r: r['2P_MADE']/r['2P_ATTEMPTS'] if r['2P_ATTEMPTS'] != 0 else np.nan, axis=1)
detailed_data['3P_PERCENT'] = detailed_data.apply(lambda r: r['3P_MADE']/r['3P_ATTEMPTS'] if r['3P_ATTEMPTS'] != 0 else np.nan, axis=1)

detailed_data['2P_POINTS'] = detailed_data.apply(lambda r: r['2P_MADE'] * 2, axis=1)
detailed_data['3P_POINTS'] = detailed_data.apply(lambda r: r['3P_MADE'] * 3, axis=1)
detailed_data['PLAYER_MINUTES_REGULAR'] = 0
detailed_data['PLAYER_MINUTES_CRUNCH'] = 0
data = data[data['GAME_ID'].isin(list(detailed_data['GAME_ID'].unique()))] #getting only relevant games to this project based on players.
data['SCOREMARGIN'] = data['SCOREMARGIN'].apply(lambda x: float(str(x).replace('TIE','0')))
#Seperating clutch situations from non clutch situations
data['CRUNCH'] = 0
for i, r in data.iterrows():
    if r['PERIOD'] < 4:
        continue
    if pd.isnull(r['SCOREMARGIN']):
        continue
    elif abs(r['SCOREMARGIN']) < 10 and r['PERIOD'] >= 4:
        data.at[i,'CRUNCH'] = 1

data['TIME'] = data['TIME'].apply(lambda x: x/60)
game_crunch = {}
game_stops = [0,12, 24, 36, 48, 53, 58, 63, 68]

def checkinRange(time,myrange):
    if time >= myrange[0] and time <= myrange[1]:
        return True
    else:
        return False

for game in data['GAME_ID'].unique():
    mygame_df = data[data['GAME_ID'] == game]
    total_game_mins = mygame_df['TIME'].max()
    temp = mygame_df[data['CRUNCH'] == 1][['TIME','CRUNCH']]
    temp = temp.sort_values(by='TIME').reset_index(drop=True)
    if len(temp) != 0: #if there is crunch time in a game.
        game_crunch[game] = [temp['TIME'][0],temp['TIME'][len(temp)-1],total_game_mins]
        crunch = True
        crunchminutes = game_crunch[game][1] - game_crunch[game][0]
        noncrunchminutes = total_game_mins - crunchminutes        
    else: #if there is no crunch time in a game
        game_crunch[game] = [0,0,total_game_mins]
        crunch = False
        noncrunchminutes = total_game_mins
        crunchminutes = 0
    range_crunch = [game_crunch[game][0],game_crunch[game][1]]
    players_now = [x for x in players if x in mygame_df['SUB_ENTERED_PLAYER_ID'].unique() or x in mygame_df['SUB_LEAVING_PLAYER_ID'].unique()]
    for player in players_now:
        player_entrylog = []
        player_exitlog = []
        missed_time_regular = 0
        missed_time_crunch = 0
        player_exitlog = list(mygame_df[mygame_df['SUB_LEAVING_PLAYER_ID'] == player]['TIME'].unique())
        player_entrylog = list(mygame_df[mygame_df['SUB_ENTERED_PLAYER_ID'] == player]['TIME'].unique())
        player_log = sorted(player_exitlog + player_entrylog)
        done_times = []
        for ind, time in enumerate(player_log):
            if time in done_times:
                continue
            if time in player_exitlog and not checkinRange(time,range_crunch): #if the loop goes here, it means the player first left the game and left the game during regular time
                if ind < len(player_log)-1:
                    if player_log[ind + 1] in player_entrylog and not checkinRange(player_log[ind + 1],range_crunch): #if he left during regular time and entered during regular time.
                        missed_time_regular = missed_time_regular + (player_log[ind + 1]-time)
                        player_exitlog.remove(time)
                        done_times.append(time)
                        player_entrylog.remove(player_log[ind + 1])
                        done_times.append(time)
                    elif player_log[ind + 1] in player_entrylog and checkinRange(player_log[ind + 1],range_crunch): #if he left during regular time and entered during crunch time
                        missed_time_regular = missed_time_regular + (game_crunch[game][0]-time) #from the time he left to the beginning of crunch time, is the regular time missed
                        missed_time_crunch = missed_time_crunch + (player_log[ind + 1] - game_crunch[game][0]) #from the beginning of crunch time till the player entered
                        player_exitlog.remove(time)
                        done_times.append(time)
                        player_entrylog.remove(player_log[ind + 1])
                    elif player_log[ind + 1] not in player_entrylog:#player left during regular time and didn't enter again in the same quarter
                        for t in game_stops:
                            if t >= time: #find out the beginning of the next quater t/or end of the game and find how long before t the person left and didn't enter 
                                missed_time_regular = missed_time_regular + (t-time)
                                break
                        player_exitlog.remove(time)
                        done_times.append(time)
            elif time in player_exitlog and checkinRange(time,range_crunch): #if the player left during crunch time
                if ind < len(player_log)-1:
                    if player_log[ind + 1] in player_entrylog and checkinRange(player_log[ind + 1],range_crunch): #if the player left during crunch time and entered during crunch time
                        missed_time_crunch = missed_time_crunch + (game_crunch[game][1] - time) #time from when they left till end of crunch time
                        missed_time_regular = missed_time_regular + (player_log[ind + 1] - game_crunch[game][1]) #time from end of crunch time till when they entered
                        player_exitlog.remove(time)
                        done_times.append(time)
                        player_entrylog.remove(player_log[ind + 1])
                    elif player_log[ind + 1] in player_entrylog and not checkinRange(player_log[ind + 1],range_crunch): #if the player left during crunch time and reentered after crunch(unlikely)
                        missed_time_crunch = missed_time_crunch + (player_log[ind + 1] - time)
                        player_exitlog.remove(time)
                        done_times.append(time)
                        player_entrylog.remove(player_log[ind + 1])
                    elif player_log[ind + 1] not in player_entrylog: #player left during crunch time and didn't reenter the game.
                        for t in game_stops:
                            if t >= time: #find out the beginning of the next quater t/or end of the game and find how long before t the person left and didn't enter 
                                missed_time_crunch = missed_time_crunch + (t-time)
                                break
                        player_exitlog.remove(time)
                        done_times.append(time)                            
            elif time in player_entrylog and not checkinRange(time,range_crunch): #the only scenario the loop would go here is if the player didn't start or the game and if he finished a quarter and didn't start the next quarter.
                for q,t in enumerate(game_stops):
                    if t < time: #skip all previously completed quarters
                        continue
                    else:
                        missed_time_regular = missed_time_regular + (time-game_stops[q-1]) #from the start of the current quarter till the time player entered the game
                        break
                player_entrylog.remove(time)
                done_times.append(time)
            elif time in player_entrylog and checkinRange(time,range_crunch): #left before the end odf the previous quater and entered in crunch time
                for q,t in enumerate(game_stops):
                    if t < time: #skip the start of all quaters that were completed
                        continue
                    else: #the moment t > time, use the previous value of t #game_stops[q-1] is the time when new quarter started.
                        missed_time_regular = missed_time_regular + (game_crunch[game][0]-game_stops[q-1]) #from the time the new quarter started till the start of crunch time
                        missed_time_crunch = missed_time_crunch + (time - game_crunch[game][0]) #from the beginning of crunch time till the player entered the game
                        break
                

        row_ind_noncrunch = detailed_data[(detailed_data.GAME_ID == game) & (detailed_data.CRUNCH == 0) & (detailed_data.PLAYER == player_id_dict[player])].index.tolist()[0]
        detailed_data.loc[row_ind_noncrunch,'PLAYER_MINUTES_REGULAR'] = noncrunchminutes - missed_time_regular
        try:
            row_ind_crunch = detailed_data[(detailed_data.GAME_ID == game) & (detailed_data.CRUNCH == 1) & (detailed_data.PLAYER == player_id_dict[player])].index.tolist()[0]
            detailed_data.loc[row_ind_crunch,'PLAYER_MINUTES_CRUNCH'] = crunchminutes - missed_time_crunch
        except:
            bla = 0
            
                   
                

detailed_data_crunch_new = pd.DataFrame()
detailed_data_crunch = detailed_data[detailed_data['CRUNCH'] == 1]
for player in detailed_data_crunch['PLAYER'].unique():
    temp = detailed_data_crunch[detailed_data_crunch['PLAYER'] == player]
    temp['3P_PERCENT'].fillna(temp['3P_PERCENT'].mean(), inplace=True)
    temp['2P_PERCENT'].fillna(temp['2P_PERCENT'].mean(), inplace=True)
    detailed_data_crunch_new = detailed_data_crunch_new.append(temp, ignore_index=True)

detailed_data_regular = detailed_data[detailed_data['CRUNCH'] == 0]
detailed_data_regular_new = pd.DataFrame()
for player in detailed_data_regular['PLAYER'].unique():
    temp = detailed_data_regular[detailed_data_regular['PLAYER'] == player]
    temp['3P_PERCENT'].fillna(temp['3P_PERCENT'].mean(), inplace=True)
    temp['2P_PERCENT'].fillna(temp['2P_PERCENT'].mean(), inplace=True)
    detailed_data_regular_new = detailed_data_regular_new.append(temp, ignore_index=True)

    
for col in ['2P_POINTS', '3P_POINTS','BLOCKS','ASSISTS','REBOUNDS']:
    detailed_data_crunch_new[col + '_PER_MINUTE'] = detailed_data_crunch_new.apply(lambda r: r[col]/r['PLAYER_MINUTES_CRUNCH'], axis=1)

for col in ['2P_POINTS', '3P_POINTS','BLOCKS','ASSISTS','REBOUNDS']:
    detailed_data_regular_new[col + '_PER_MINUTE'] = detailed_data_regular_new.apply(lambda r: r[col]/r['PLAYER_MINUTES_REGULAR'], axis=1)


detailed_data = detailed_data_regular_new.append(detailed_data_crunch_new, ignore_index=True).sort_values(by='GAME_ID')
model_data = detailed_data[['PLAYER', 'GAME_ID','2P_PERCENT', '3P_PERCENT','2P_POINTS_PER_MINUTE', '3P_POINTS_PER_MINUTE', 'BLOCKS_PER_MINUTE', 'ASSISTS_PER_MINUTE', 'REBOUNDS_PER_MINUTE','CRUNCH']]
model_data = model_data.reset_index(drop=True)
model_data.to_feather('model_data.feather')
detailed_data.to_excel('detailed_data.xlsx',index=False)
