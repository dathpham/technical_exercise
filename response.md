# Technical Exercise


1. Understanding data quality is essential in data science, and data quality encompasses a variety of dimensions. Examine the Occurrence table and evaluate the completeness of the dataset. Use these findings to provide recommendations about what types of analysis could be conducted with this data to better understand aviation accidents and safety incidents as reported in the ASIS data.

![image](https://github.com/dathpham/technical_exercise/blob/main/question1.png)

The Occurrence table is a 52742 rows x 242 columns table that contains a lot of null rows which has been counted as above. The column 'SafetyCommIssuedEnum_DisplayEng' and 'SafetyCommIssuedEnum_DisplayFre' have the most null with 52742. Type of analysis could be conducted to better understand aviation accidents and safety incidents :
- Assess the extent and pattern of missing values to understand if they are random or systematic :
    -- Calculate the percentage of missing values for each variable
    -- Determine the missing values are random or systematic. If it is systematic, determine the root cause.
    -- Examine the pattern of missing values: look for any trends or relationships between missing values and other variables. 

- Conduct deletion or data imputation on the missing data.
- Conduct the sensitivity analysis on the impact of missing data to the safety and fatality of the accident results. Assess how the results would change if the missing data mechanism was different.




4. Develop a model that predicts the probability of surviving a safety incident. What are the key factors
associated with survivability? What are the strengths and weaknesses of your analysis and what would
your next steps be with this model?

![image](https://github.com/dathpham/technical_exercise/blob/main/Question4.png)
![image](https://github.com/dathpham/technical_exercise/blob/main/Question4_Feature_Importance.png)








5. What ICAO event categories are most common at Canadian airports? Is there any trend or pattern evident
in these Canadian events?


![image](https://github.com/dathpham/technical_exercise/blob/main/Question5.PNG)
