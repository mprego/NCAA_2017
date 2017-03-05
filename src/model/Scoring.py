#Scores model using a tournament style scoring method

import pandas as pd
import numpy as np
import os

class Scorer(object):

    def __init__(self, slots, seeds, results, model, features):
        self.slots = slots
        self.seeds = seeds
        self.results = results
        self.model = model
        self.features = features

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
        return slots_teams_0

    #Method for creating SS for a round
    def create_round_ss(self, rnd, prev_results):
        slots_r = self.slots[self.slots['Round']==('R'+str(rnd))]

        #Adds winners from previous round to this round
        slots_r = slots_r.merge(prev_results[['Season', 'Slot', 'Wteam']], left_on=['Strongseed', 'Season'], right_on=['Slot', 'Season'], how='left')
        slots_r = slots_r.rename(index=str, columns={'Wteam':'StrongTeam_x'})
        slots_r = slots_r.merge(prev_results[['Season', 'Slot', 'Wteam']], left_on=['Weakseed', 'Season'], right_on=['Slot', 'Season'], how='left')
        slots_r = slots_r.drop(['Slot', 'Slot_y'], 1).rename(index=str, columns={'Wteam':'WeakTeam_x', 'Slot_x':'Slot'})

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

    def score_round(self, round_data, pred):
        if pred==False:
            scored_round = round_data.merge(self.results[['Season', 'matchup', 'Wteam']], on=['Season', 'matchup'])
        else:
            ss = round_data.merge(self.features, left_on=['Season', 'StrongTeam'], right_on=['Season', 'Team'], how='inner')
            ss = ss.merge(self.features, left_on=['Season', 'WeakTeam'], right_on=['Season', 'Team'], suffixes=('_W', '_L'), how='inner')

            ss = ss[list(self.model.input_cols)].sort_index(axis=1)
            scored_round = round_data.copy()
            scored_round['Wteam'] = [s if o>0 else w for o,s,w in zip(self.model.get_pred(ss), scored_round['StrongTeam'], scored_round['WeakTeam'])]
        return scored_round

    # Simulator method

    def simulate_tournament(self, pred):
        r0_data = self.create_round_0()
        r1_ss = self.create_round_ss(1, r0_data)
        r1_scored = self.score_round(r1_ss, pred)
        r2_ss = self.create_round_ss(2, r1_scored)
        r2_scored = self.score_round(r2_ss, pred)
        r3_ss = self.create_round_ss(3, r2_scored)
        r3_scored = self.score_round(r3_ss, pred)
        r4_ss = self.create_round_ss(4, r3_scored)
        r4_scored = self.score_round(r4_ss, pred)
        r5_ss = self.create_round_ss(5, r4_scored)
        r5_scored = self.score_round(r5_ss, pred)
        r6_ss = self.create_round_ss(6, r5_scored)
        r6_scored = self.score_round(r6_ss, pred)

        return [r1_scored, r2_scored, r3_scored, r4_scored, r5_scored, r6_scored]


    #Scores predictions using tournament style scoring system
    def score_model(self):
        actuals = self.simulate_tournament(False)
        pred = self.simulate_tournament(True)
        score = 0
        for idx in range(6):
            act = actuals[idx][['Season', 'Slot', 'Wteam']]
            pre = pred[idx][['Season', 'Slot', 'Wteam']]
            comb = act.merge(pre, on=['Season', 'Slot'])
            comb['Correct'] = [1 if a==b else 0 for a,b in zip(comb['Wteam_x'], comb['Wteam_y'])]
            pts = np.sum(comb['Correct'])
            score += pts * (2**idx)
        score = 1.0 * score / len(set(self.slots['Season']))
        return score
