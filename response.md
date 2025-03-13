# Technical Exercise


1. Understanding data quality is essential in data science, and data quality encompasses a variety of dimensions. Examine the Occurrence table and evaluate the completeness of the dataset. Use these findings to provide recommendations about what types of analysis could be conducted with this data to better understand aviation accidents and safety incidents as reported in the ASIS data.

![image](https://github.com/dathpham/technical_exercise/blob/main/question1.png)

The Occurrence table is a 52742 rows x 242 columns table that contains a lot of null rows which has been counted as above. The column 'SafetyCommIssuedEnum_DisplayEng' and 'SafetyCommIssuedEnum_DisplayFre' have the most null with 52742. Type of analysis could be conducted to better understand aviation accidents and safety incidents :
- Assess the extent and pattern of missing values to understand if they are random or systematic :
    * Calculate the percentage of missing values for each variable
    * Determine the missing values are random or systematic. If it is systematic, determine the root cause.
    * Examine the pattern of missing values: look for any trends or relationships between missing values and other variables. 

- Conduct deletion or data imputation on the missing data.
- Conduct the sensitivity analysis on the impact of missing data to the safety and fatality of the accident results. Assess how the results would change if the missing data mechanism was different.



4. Develop a model that predicts the probability of surviving a safety incident. What are the key factors
associated with survivability? What are the strengths and weaknesses of your analysis and what would
your next steps be with this model?

XGboost Classifier is used to classify the incident as fatality or not. 

Features used are : 

'Visibility','WeatherPhenomenaTypeID_DisplayEng','WeatherPhenomenaIntensityID_DisplayEng','EngineTypeID_DisplayEng','PropellerTypeID_DisplayEng','CombustibleID_DisplayEng'

       
Class features in the above list are 

'WeatherPhenomenaTypeID_DisplayEng','WeatherPhenomenaIntensityID_DisplayEng','EngineTypeID_DisplayEng','PropellerTypeID_DisplayEng','CombustibleID_DisplayEng'

and they will be one-hot-encoded. 

The complete features list after one hot encoding are :

       'Visibility',
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
       'CombustibleID_DisplayEng_OTHER'

The output class is 'TotalFatalCount' that is mapped to 'survivality' with 0 if there is non-zero fatal count or else the 'survivality' is 1. The model is trained with 10 estimators, max depth of 2 and 0.1 learning rate.
The accuracy on training set is  0.968 and test set accuracy is 0.57. The model is having overfitting issue. The features with most importance are : CombustibleID_DisplayEng_FUEL and the Visibility. Altering value in these features will impact the result the most.
The weakness of my analysis :
- Overfitting
- Lack meaningful system metric as most of them contain NULL

![image](https://github.com/dathpham/technical_exercise/blob/main/Question4_r.PNG)
![image](https://github.com/dathpham/technical_exercise/blob/main/Question4_Feature_Importance1.png)



5. What ICAO event categories are most common at Canadian airports? Is there any trend or pattern evident
in these Canadian events?
The most common event happens at Canadian airports is the SYSTEM/COMPONENT FAILURE OR MALFUNCTION (NON-POWERPLANT) (SCF–NP) with 2623 incidents. The second most incident is SYSTEM/COMPONENT FAILURE OR MALFUNCTION (POWERPLANT) (SCF–PP). The trend or pattern is incidents are more related to system issue or component failure.

![image](https://github.com/dathpham/technical_exercise/blob/main/Question5.PNG)


