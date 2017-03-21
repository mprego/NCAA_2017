#Scores model using a tournament style scoring method

import pandas as pd
import numpy as np
import os
import random

class Scorer(object):

    def __init__(self, features):
        self.slots = None
        self.seeds = None
        self.results = None
        self.model = None
        self.features = features

    def set_variables(self, slots, seeds, results, model):
        self.slots = slots
        self.seeds = seeds
        self.results = results
        self.model = model
        self.slots['Round'] = ['R0' if x[0:1] in ['W', 'X', 'Y', 'Z'] else x[0:2] for x in self.slots['Slot']]

    #Method for creating round 0 games
    def create_round_0(self):
        #Adds Round 0 winners
        slots_0 = self.slots[self.slots['Round']=='R0']

        #joins slots with team names
        slots_teams_0 = slots_0.merge(self.seeds, left_on=['Strongseed', 'Season'], right_on=['Seed', 'Season'], how='left')
        slots_teams_0 = slots_teams_0.rename(index=str, columns={'Team': 'StrongTeam'}).drop('Seed', 1)

        slots_teams_0 = slots_teams_0.merge(self.seeds, left_on=['Weakseed', 'Season'], right_on=['Seed', 'Season'], how='left')
        slots_teams_0 = slots_teams_0.rename(index=str, columns={'Team': 'WeakTeam'}).drop('Seed', 1)

        #Adds matchup field to slots and results
        slots_teams_0['matchup'] = [str(a)+'_'+str(b) if a<b else str(b)+'_'+str(a) for a,b in zip(slots_teams_0['StrongTeam'], slots_teams_0['WeakTeam'])]
        self.results['matchup'] = [str(a)+'_'+str(b) if a<b else str(b)+'_'+str(a) for a,b in zip(self.results['Wteam'], self.results['Lteam'])]

        #Adds results to round 0 games
        slots_teams_0 = slots_teams_0.merge(self.results, on=['Season', 'matchup'])
        slots_teams_0 = slots_teams_0.rename(index=str, columns={'Wteam':'Oteam'})
        return slots_teams_0

    #Method for creating SS for a round
    def create_round_ss(self, rnd, prev_results):
        slots_r = self.slots[self.slots['Round']==('R'+str(rnd))]

        #Adds winners from previous round to this round
        slots_r = slots_r.merge(prev_results[['Season', 'Slot', 'Oteam']], left_on=['Strongseed', 'Season'], right_on=['Slot', 'Season'], how='left')
        slots_r = slots_r.rename(index=str, columns={'Oteam':'StrongTeam_x'})
        slots_r = slots_r.merge(prev_results[['Season', 'Slot', 'Oteam']], left_on=['Weakseed', 'Season'], right_on=['Slot', 'Season'], how='left')
        slots_r = slots_r.drop(['Slot', 'Slot_y'], 1).rename(index=str, columns={'Oteam':'WeakTeam_x', 'Slot_x':'Slot'})

        #joins slots with team names
        slots_teams_r = slots_r.merge(self.seeds, left_on=['Strongseed', 'Season'], right_on=['Seed', 'Season'], how='left')
        slots_teams_r = slots_teams_r.rename(index=str, columns={'Team': 'StrongTeam'}).drop('Seed', 1)

        slots_teams_r = slots_teams_r.merge(self.seeds, left_on=['Weakseed', 'Season'], right_on=['Seed', 'Season'], how='left')
        slots_teams_r = slots_teams_r.rename(index=str, columns={'Team': 'WeakTeam'}).drop('Seed', 1)

        #combines Strongteam and Weakteam columns
        slots_teams_r['StrongTeam'] = slots_teams_r['StrongTeam'].fillna(slots_teams_r['StrongTeam_x'])
        slots_teams_r['WeakTeam'] = slots_teams_r['WeakTeam'].fillna(slots_teams_r['WeakTeam_x'])
        slots_teams_r['StrongTeam'] = slots_teams_r['StrongTeam'].astype('int')
        slots_teams_r['WeakTeam'] = slots_teams_r['WeakTeam'].astype('int')
        slots_teams_r = slots_teams_r.drop(['StrongTeam_x', 'WeakTeam_x'], 1)

        #Adds matchup column
        slots_teams_r['matchup'] = [str(a)+'_'+str(b) if a<b else str(b)+'_'+str(a) for a,b in zip(slots_teams_r['StrongTeam'], slots_teams_r['WeakTeam'])]

        return slots_teams_r

    #Score round
    #Function get a given SS and scores it using real data (may need round info, but matchup should suffice)
    #Also, allow a custom scorer to look at SS to determine winner

    def score_round(self, round_data, pred, interactions=None, probabilistic=False):
        if pred==False:
            scored_round = round_data.merge(self.results[['Season', 'matchup', 'Wteam']], on=['Season', 'matchup'])
            scored_round = scored_round.rename(index=str, columns={'Wteam':'Oteam'})
        else:
            #renames
            round_data_mod = round_data.rename(index=str, columns={'StrongTeam':'Wteam', 'WeakTeam':'Lteam'})
            ss = self.make_ss(round_data_mod, interactions, False)

            # ss = round_data.merge(self.features, left_on=['Season', 'StrongTeam'], right_on=['Season', 'Team'], how='inner')
            # ss = ss.merge(self.features, left_on=['Season', 'WeakTeam'], right_on=['Season', 'Team'], suffixes=('_W', '_L'), how='inner')

            ss = ss[list(self.model.input_cols)].sort_index(axis=1)
            scored_round = round_data_mod.copy()
            if probabilistic == False:
                scored_round['Oteam'] = [s if o>0 else w for o,s,w in zip(self.model.get_pred(ss), scored_round['Wteam'], scored_round['Lteam'])]
            else:
                # scored_round['prob'] = self.model.get_prob(ss)
                scored_round['Oteam'] = [s if p<random.random() else w for p,s,w in zip(self.model.get_prob(ss), scored_round['Wteam'], scored_round['Lteam'])]

        return scored_round

    # Simulator method

    def simulate_tournament(self, pred, interactions=None, probabilistic=False):
        r0_data = self.create_round_0()
        r1_ss = self.create_round_ss(1, r0_data)
        r1_scored = self.score_round(r1_ss, pred, interactions, probabilistic)
        r2_ss = self.create_round_ss(2, r1_scored)
        r2_scored = self.score_round(r2_ss, pred, interactions, probabilistic)
        r3_ss = self.create_round_ss(3, r2_scored)
        r3_scored = self.score_round(r3_ss, pred, interactions, probabilistic)
        r4_ss = self.create_round_ss(4, r3_scored)
        r4_scored = self.score_round(r4_ss, pred, interactions, probabilistic)
        r5_ss = self.create_round_ss(5, r4_scored)
        r5_scored = self.score_round(r5_ss, pred, interactions, probabilistic)
        r6_ss = self.create_round_ss(6, r5_scored)
        r6_scored = self.score_round(r6_ss, pred, interactions, probabilistic)

        return [r1_scored, r2_scored, r3_scored, r4_scored, r5_scored, r6_scored]
        # return [r1_scored, r2_scored, r3_scored, r4_scored, r5_scored, r6_scored, r1_ss, r2_ss, r3_ss, r4_ss, r5_ss, r6_ss]


    #Scores predictions using tournament style scoring system
    def score_model(self, interactions=None, trials=1):
        # diag = []

        actuals = self.simulate_tournament(False)
        score = 0
        if trials>1:
            aggregate = pd.DataFrame()
            for trial in range(trials):
                pred_trial = self.simulate_tournament(True, interactions, True)
                for rnd in pred_trial:
                    aggregate = aggregate.append(rnd[['Season', 'Slot', 'Oteam', 'matchup']])
            aggregate_idx = aggregate.groupby(['Season', 'Slot', 'Oteam']).count().reset_index()
            pred = pd.DataFrame()
            idx = 0
            for s in set(aggregate_idx['Season']):
            # for s in [2015]:
                for sl in set(aggregate_idx['Slot']):
                    pred.set_value(idx, 'Season', int(s))
                    pred.set_value(idx, 'Slot', sl)
                    pred.set_value(idx, 'Round', sl[0:2])
                    options = aggregate_idx[(aggregate_idx['Season']==s) & (aggregate_idx['Slot']==sl)]
                    options_sorted = options.sort_values(by='matchup')
                    # diag.append(options_sorted)
                    pred.set_value(idx, 'Oteam', options_sorted.tail(1)['Oteam'].values[0])
                    idx += 1
                pred['Oteam'] = [int(a) for a in pred['Oteam']]
                for idx2 in range(6):
                    act = actuals[idx2][['Season', 'Slot', 'Oteam']]
                    pre = pred[pred['Round']==('R'+str(idx2+1))][['Season', 'Slot', 'Oteam']]
                    # diag.append(pre)
                    # return diag
                    comb = act.merge(pre, on=['Season', 'Slot'])
                    comb['Correct'] = [1 if a==b else 0 for a,b in zip(comb['Oteam_x'], comb['Oteam_y'])]
                    pts = np.sum(comb['Correct'])
                    score += pts * (2**idx2)
                    # print pts, (idx2+1), score
                score = 1.0 * score / len(set(self.slots['Season']))
            return score

        else:
            pred = self.simulate_tournament(True, interactions)

            for idx in range(6):
                act = actuals[idx][['Season', 'Slot', 'Oteam']]
                pre = pred[idx][['Season', 'Slot', 'Oteam']]
                comb = act.merge(pre, on=['Season', 'Slot'])
                comb['Correct'] = [1 if a==b else 0 for a,b in zip(comb['Oteam_x'], comb['Oteam_y'])]
                pts = np.sum(comb['Correct'])
                score += pts * (2**idx)
            score = 1.0 * score / len(set(self.slots['Season']))
            return score

    #Given a DF of games of wteam, lteam, and season, returns attached variables needed for model
    def make_ss(self, games, interactions=None, randomize=True):
        #randomizes wins and losses
        if randomize==True:
            games = self.randomize_ss(games)

        #joins data with features
        ss = games.merge(self.features, left_on=['Season', 'Wteam'], right_on=['Season', 'Team'], how='inner')
        ss = ss.merge(self.features, left_on=['Season', 'Lteam'], right_on=['Season', 'Team'], suffixes=('_A', '_B'), how='inner')
        ss = ss.drop(['Lteam', 'Wteam'], 1)

        #adds interaction effect, if specified
        if interactions is not None:
            for combo in interactions:
                a_col = combo[0]
                b_col = combo[1]
                inter_col = a_col + '_' + b_col
                ss[inter_col] = ss[a_col] * ss[b_col]

        if self.model is not None:
            ss = ss[list(self.model.input_cols)]

        return ss.sort_index(axis=1)

    #Randomly swithces W and L teams
    def randomize_ss(self, ss):
        random_picks_to_switch = random.sample(ss.index, len(ss.index)/2)
        replace_wl = ss.ix[random_picks_to_switch,:].copy()
        replace_wl_renamed = pd.DataFrame()
        if 'Wscore' in ss.columns.values:
            replace_wl_renamed['Wscore'] = replace_wl['Lscore']
            replace_wl_renamed['Lscore'] = replace_wl['Wscore']
        replace_wl_renamed['Wteam'] = replace_wl['Lteam']
        replace_wl_renamed['Lteam'] = replace_wl['Wteam']
        replace_wl_renamed['Season'] = replace_wl['Season']
        # replace_w1 = replace_wl.rename(index=str, columns={'Wscore':'score', 'Wteam_Name':'team_Name'})
        # replace_w1 = replace_wl.rename(index=str, columns={'Lscore':'Wscore', 'Lteam_Name':'Wteam_Name'})
        # replace_w1 = replace_wl.rename(index=str, columns={'score':'Lscore', 'team_Name':'Lteam_Name'})
        ss = ss.drop(random_picks_to_switch)
        ss = ss.append(replace_wl_renamed)
        if 'Wscore' in ss.columns.values:
            ss = ss[['Lscore', 'Wscore', 'Wteam', 'Lteam', 'Season']].reset_index(drop=True)
        else:
            ss = ss[['Wteam', 'Lteam', 'Season']].reset_index(drop=True)
        return ss

    #Given a list of interactions, predicts tournament and returns actuals and predictions
    def get_tourney_pred(self, interactions=None, trials=1, just_pred=False):
        if just_pred==True:
            pred = self.simulate_tournament(True, interactions)
            pred_data = pd.DataFrame()
            for y in pred:
                pred_data = pred_data.append(y).reset_index(drop=True)
            return pred
        actuals = self.simulate_tournament(False)
        actuals_data = pd.DataFrame()
        for x in actuals:
            actuals_data = actuals_data.append(x).reset_index(drop=True)
        if trials>1:
            aggregate = pd.DataFrame()
            for trial in range(trials):
                pred_trial = self.simulate_tournament(True, interactions, True)
                for rnd in pred_trial:
                    aggregate = aggregate.append(rnd[['Season', 'Slot', 'Oteam', 'matchup']])
            aggregate_idx = aggregate.groupby(['Season', 'Slot', 'Oteam']).count().reset_index()
            pred = pd.DataFrame()
            idx = 0
            for s in set(aggregate_idx['Season']):
                for sl in set(aggregate_idx['Slot']):
                    pred.set_value(idx, 'Season', s)
                    pred.set_value(idx, 'Slot', sl)
                    pred.set_value(idx, 'Round', sl[0:2])
                    options = aggregate_idx[(aggregate_idx['Season']==s) & (aggregate_idx['Slot']==sl)]
                    options_sorted = options.sort_values(by='matchup').reset_index()
                    pred.set_value(idx, 'Oteam', options['Oteam'].tail(1).values[0])
                    idx += 1
            pred_data = pred
        else:
            pred = self.simulate_tournament(True, interactions)
            pred_data = pd.DataFrame()
            for y in pred:
                pred_data = pred_data.append(y).reset_index(drop=True)

        return (actuals_data, pred_data)
