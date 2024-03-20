def clean_seeds(seed):
    res = seed[1:]

    if len(res) > 2:
        res = res[:-1]

    return int(res)

def build_seeds(gender, csvs):
    seeds = csvs["{}NCAATourneySeeds".format(gender)] 
    seeds['Seed'] = seeds['Seed'].apply(clean_seeds)
    seeds = seeds.drop('Season', axis=1)
    seeds = seeds.groupby(by='TeamID').mean()
    
    return seeds

def build_rankings(gender, csvs):
    rankings = csvs["{}MasseyOrdinals".format(gender)]
    rankings = rankings.drop(['SystemName', 'RankingDayNum'], axis=1)
    rankings = rankings.groupby(by='TeamID').mean()
    rankings = rankings.drop('Season', axis=1)

    return rankings

