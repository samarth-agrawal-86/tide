import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from joblib import load
from itertools import combinations

SEED = 2021

def generate_prediction(data, threshold, max_ranks, features):
    df_orig = data.copy()
    df = data.copy()

    # Feature Engineering
    df['Mapping_Desc_Pred_DiffPred_Sum'] = df['DateMappingMatch'] + df['AmountMappingMatch']  +df['TimeMappingMatch'] \
                    + df['DescriptionMatch'] + +df['ShortNameMatch'] \
                    + df['PredictedNameMatch']+ df['PredictedAmountMatch']+ df['PredictedTimeCloseMatch'] \
                    + df['DifferentPredictedTime'] + df['DifferentPredictedDate']

    df['Mapping_Desc_Sum'] = df['DateMappingMatch'] + df['AmountMappingMatch']  +df['TimeMappingMatch'] \
                    + df['DescriptionMatch'] + +df['ShortNameMatch']

    df['Mapping_Desc_Pred_Sum'] = df['DateMappingMatch'] + df['AmountMappingMatch']  +df['TimeMappingMatch'] \
                    + df['DescriptionMatch'] + +df['ShortNameMatch'] \
                    + df['PredictedNameMatch']+ df['PredictedAmountMatch']+ df['PredictedTimeCloseMatch']

    df['NegDifferentPredictedTime'] = -1 * df['DifferentPredictedTime']   
    df['NegDifferentPredictedDate'] = -1 * df['DifferentPredictedDate'] 


    comb_vars = ['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch', 'NegDifferentPredictedTime', 
                'TimeMappingMatch', 'PredictedNameMatch', 'ShortNameMatch', 'NegDifferentPredictedDate',
                'PredictedAmountMatch', 'PredictedTimeCloseMatch', ]

    # Interaction features of 2 variables
    comb2 = combinations(comb_vars,2)

    comb2_vars = []
    for i in list(comb2):
        var = i[0] + '_' + i[1]
        comb2_vars.append(var)

        df[var] = df[i[0]] + df[i[1]]



    # Interaction features of 3 variables
    comb3 = combinations(comb_vars,3)

    comb3_vars = []
    for i in list(comb3):
        var = i[0] + '_' + i[1] + '_' + i[2]
        comb3_vars.append(var)

        df[var] = df[i[0]] + df[i[1]] + df[i[2]]

    
    # Keep only relevant features
    df = df[features]
    
    lgb_model2 = LGBMClassifier()
    lgb_model2 = load('lgb_model2.joblib')
    
    y = pd.Series(lgb_model2.predict_proba(df)[:,1])
    y.index = df.index

    df_pred = pd.concat([df_orig, y], axis=1)
    df_pred.rename(columns = {0:'pred_prob'}, inplace=True)
    df_pred = df_pred.sort_values(by= ['receipt_id','pred_prob'], ascending=[True, False])
    df_pred['pred_target'] = np.where(df_pred['pred_prob']>=threshold, 1, 0)
    df_pred['pred_prob_rank'] = df_pred.groupby('receipt_id')['pred_prob'].rank(method='min', ascending=False)
    df_pred['pred_prob_rank'] = df_pred['pred_target'] * df_pred['pred_prob_rank']
    df_pred['pred_prob_rank'] = np.where(df_pred['pred_prob_rank']>max_ranks, 99, df_pred['pred_prob_rank'])
    df_pred.drop(columns = ['pred_prob','pred_target'], inplace=True)
    
    del df
    del df_orig
    
    return df_pred