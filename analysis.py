import pandas as pd
import matplotlib as plt
import sklearn
import os, datetime
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

cwd = os.getcwd()
occurence_dir = cwd + "\\data\\ASISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv"
aircraft_dir = cwd + "\\data\\ASISdb_MDOTW_VW_AIRCRAFT_PUBLIC.csv"
injury_dir = cwd + "\\data\\ASISdb_MDOTW_VW_INJURIES_PUBLIC.csv"
event_dir = cwd + "\\data\\ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv"
surviv_dir = cwd + "\\data\\ASISdb_MDOTW_VW_SURVIVABILITY_PUBLIC.csv"
OC_col = ['OccID','AirportID','OccDate',
          'AirportID_CountryID_DisplayEng','Distance',
          'ICAO','ICAO_DisplayEng','OccIncidentTypeID_DisplayEng',
          'TotalFatalCount','NoAircraftInvolved',
          'WeatherPhenomenaTypeID_DisplayEng','Visibility',
          'SwPeakGus','WeatherPhenomenaIntensityID_DisplayEng',
          ]
AC_col = ['occid','EngineTypeID','EngineTypeID_DisplayEng',
          'AircraftID','PropellerTypeID','PropellerTypeID_DisplayEng',
          'RotorSystemID','RotorSystemID_DisplayEng','CombustibleID','CombustibleID_DisplayEng']
SUR_col = ['OccID','SurvivableEnum','SurvivableEnum_DisplayEng']
df_oc = pd.read_csv(occurence_dir,low_memory=False)
df_ac = pd.read_csv(aircraft_dir,low_memory=False)
df_su = pd.read_csv(surviv_dir,low_memory=False)

# Redirect stdout to a file
with open(cwd+'\\outputs\\Question1_NaN_count.csv', 'w') as f:
    sys.stdout = f
    print("Count of NaN in each column of Occurrence Dataset:")
    print(df_oc.isna().sum())


df_oc =df_oc[OC_col].set_index('OccID')
df_ac = df_ac[AC_col]
df_ac.columns = ['OccID','EngineTypeID','EngineTypeID_DisplayEng',
          'AircraftID','PropellerTypeID','PropellerTypeID_DisplayEng',
          'RotorSystemID','RotorSystemID_DisplayEng','CombustibleID','CombustibleID_DisplayEng']
df_ac=df_ac.set_index('OccID')
df_su = df_su[SUR_col].set_index('OccID')
df =df_oc.join(df_ac,how = "inner").join(df_su,how = "inner")

# Save the original stdout
original_stdout = sys.stdout

features = ['Visibility','WeatherPhenomenaTypeID_DisplayEng','WeatherPhenomenaIntensityID_DisplayEng','EngineTypeID_DisplayEng','PropellerTypeID_DisplayEng','CombustibleID_DisplayEng']

target = ['TotalFatalCount','SurvivableEnum']
df =df[features+target]
df=df.dropna()
onehot_col=['WeatherPhenomenaTypeID_DisplayEng','WeatherPhenomenaIntensityID_DisplayEng','EngineTypeID_DisplayEng','PropellerTypeID_DisplayEng','CombustibleID_DisplayEng']
for i in onehot_col:

    for j in df[i].unique():
        name_col = i +'_'+str(j)
        df[name_col]=0
            
        ind = df[df[i]==j].index
        df.loc[ind,name_col]=1

df = df.drop(columns=onehot_col)

df['suvivability']=np.nan
for i in df['SurvivableEnum'].unique():
    if i == 1:
        idx =df[df['SurvivableEnum']==i].index
        df.loc[idx,'suvivability'] = 1
    elif i == 2:
        idx =df[df['SurvivableEnum']==i].index
        df.loc[idx,'suvivability'] = 0        

df=df.dropna()
features = ['Visibility',
       'WeatherPhenomenaTypeID_DisplayEng_PRECIPITATION',
       'WeatherPhenomenaTypeID_DisplayEng_OTHER PHENOMENA',
       'WeatherPhenomenaTypeID_DisplayEng_TURBULENCE',
       'WeatherPhenomenaTypeID_DisplayEng_WINDSHEAR/MICROBURST',
       'WeatherPhenomenaTypeID_DisplayEng_ICING',
       'WeatherPhenomenaIntensityID_DisplayEng_LIGHT',
       'WeatherPhenomenaIntensityID_DisplayEng_UNKNOWN',
       'WeatherPhenomenaIntensityID_DisplayEng_SEVERE',
       'WeatherPhenomenaIntensityID_DisplayEng_MODERATE',
       'WeatherPhenomenaIntensityID_DisplayEng_NIL',
       'EngineTypeID_DisplayEng_TURBO PROP',
       'EngineTypeID_DisplayEng_RECIPROCATING',
       'EngineTypeID_DisplayEng_TURBO SHAFT',
       'PropellerTypeID_DisplayEng_CONSTANT SPEED',
       'PropellerTypeID_DisplayEng_FIXED PITCH',
       'PropellerTypeID_DisplayEng_CONSTANT SPEED FULL FEATHERING',
       'PropellerTypeID_DisplayEng_VARIABLE PITCH',
       'PropellerTypeID_DisplayEng_REVERSIBLE',
       'CombustibleID_DisplayEng_FUEL', 'CombustibleID_DisplayEng_UNKNOWN',
       'CombustibleID_DisplayEng_OIL', 'CombustibleID_DisplayEng_HYDRAULIC',
       'CombustibleID_DisplayEng_AIRCRAFT MATERIAL',
       'CombustibleID_DisplayEng_OTHER']
train_data = df.iloc[:1000]
test_data =  df.iloc[1000:]
X_train, X_test, y_train, y_test = train_data[features], test_data[features], train_data['suvivability'], test_data['suvivability']

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=3, learning_rate=0.1,

    max_depth=2, random_state=0).fit(X_train, y_train)

testsetscore = clf.score(X_test, y_test)
trainsetscore = clf.score(X_train, y_train)
with open(cwd+'\\outputs\\Question4_classifier_accuracy.csv', 'w') as f:
    sys.stdout = f
    print("GradientBoostingClassifier accuracy on testset:",testsetscore )
    print("GradientBoostingClassifier accuracy on trainset:",trainsetscore )	
	
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(features)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(features)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.savefig(cwd+'\\outputs\\Question4_Feature_Importance.png')


# Save the original stdout
original_stdout = sys.stdout

# Redirect stdout to a file
with open(cwd+'\\outputs\\Question5_ICAO_count.csv', 'w') as f:
    sys.stdout = f
    print("Count of ICAO Events in Canadian Airport:")
    
    print(df_oc[df_oc['AirportID_CountryID_DisplayEng']=='CANADA'][['ICAO_DisplayEng']].value_counts())



























