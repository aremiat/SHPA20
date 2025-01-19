import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("all_data.csv", sep=";")
df_filtered = data[data['ITR'] <= 2.5]

df_filtered = df_filtered[df_filtered['SDG_07_NET_ALIGNMENT_SCORE'] >= 2]

criteria = ['Annualized return 20Y', 'Sharpe 20Y']
scaler = StandardScaler()

# Normalisation des critères dans df_filtered
df_filtered[criteria] = scaler.fit_transform(df_filtered[criteria])
df_filtered['score'] = df_filtered[criteria].mean(axis=1)
top_20_best_in_class = df_filtered.sort_values(by='score', ascending=False).head(20)
itr_average = top_20_best_in_class['ITR'].mean()

while itr_average >= 2:
    eligible_assets = df_filtered[df_filtered['ITR'] < 2]
    new_top_20_best_in_class = eligible_assets.sort_values(by='score', ascending=False).head(20)
    itr_average = new_top_20_best_in_class['ITR'].mean()
    if itr_average < 2:
        top_20_best_in_class = new_top_20_best_in_class


print("Top 20 Best in Class avec ITR < 2 :")
print(top_20_best_in_class)
print(f"Moyenne ITR : {itr_average}")

total_score = top_20_best_in_class['score'].sum()
top_20_best_in_class['weight'] = top_20_best_in_class['score'] / total_score
print(top_20_best_in_class[['ticker', 'name', 'score', 'weight']])

itr_global = (top_20_best_in_class['ITR'] * top_20_best_in_class['weight']).sum()

print("ITR global recalculé : ", itr_global)
print(top_20_best_in_class[['ticker', 'Annualized return 20Y','Sharpe 20Y','score', 'weight']])

if itr_global > 2:
    normalisation_factor = 2 / itr_global
    top_20_best_in_class['weight'] = top_20_best_in_class['weight'] * normalisation_factor
    itr_global_normalised = (top_20_best_in_class['ITR'] * top_20_best_in_class['weight']).sum()
    print("ITR global après normalisation : ", itr_global_normalised)
else:
    print("L'ITR global est déjà inférieur ou égal à 2.")

final_df = top_20_best_in_class[['ticker', 'name','Annualized return 20Y','Sharpe 20Y', 'score', 'weight']]
final_df.to_csv("fund_composition.csv", sep=";", index=False)