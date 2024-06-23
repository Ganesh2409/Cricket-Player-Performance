import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
%matplotlib inline


# # Reading The DATA 

batsman_df = pd.read_csv('Data/Batsman_Data.csv')

batsman_df.head()


bowler_df = pd.read_csv('Data/Bowler_data.csv')


bowler_df

ground_avg_df = pd.read_csv('Data/Ground_Averages.csv')

match_results_df = pd.read_csv('Data/ODI_Match_Results.csv')

match_total_df = pd.read_csv('Data/ODI_Match_Totals.csv')

players_df = pd.read_csv('Data/WC_players.csv')

players_df.head()


# ### Checking the Null values and info about the data

batsman_df.isnull().sum()


  #[17]:


batsman_df.info()


  #[18]:


bowler_df.isnull().sum()


  #[19]:


bowler_df.info()


  #[20]:


ground_avg_df.isnull().sum()


  #[21]:


ground_avg_df.info()


  #[22]:


match_results_df.isnull().sum()


match_results_df.info()


match_total_df.isnull().sum()



match_total_df.info()



players_df.isnull().sum()


players_df.info()


df = [batsman_df,bowler_df,ground_avg_df,match_results_df,match_total_df,players_df]


  #[29]:


def dimensions(df):
    for i in df:
        print(i.shape)


  #[30]:


dimensions(df)


  #[31]:


batsman_df.head()


# ### Splitting The Year into Its Corresponding month and day and year columns

  #[32]:


# As our data set has start date or date column so lets first seaprate them into day, month and year wise and drop the orignal one
def seperate_date(df):
    for i in range(len(df)):
        if 'Start Date' in df[i].columns:
            df[i]['Year']=pd.to_datetime(df[i]['Start Date'])
            df[i]['Month']=df[i]['Year'].apply(lambda x:x.month) # Extracting Month
            df[i]['Day']=df[i]['Year'].apply(lambda x:x.day)  # Extracting day
            df[i]['year']=df[i]['Year'].apply(lambda x:x.year)  # Extracting year
            print(df[i].head())
            print("-------------------------")
        else:
            print("DataFrame", i, "does not have 'Start Date' column")
            print("-------------------------")


  #[33]:


seperate_date(df)


  #[34]:


batsman_df.head(5)


  #[35]:


bowler_df.head(5)


# ##### Drop irrelavent columns from the datasets
# 

  #[36]:


def drop_irrelevant(df): 
    for i in range(len(df)): 
        columns_to_drop = ['Unnamed: 0', 'Start Date', 'Year']
        irrelevant_columns = [col for col in columns_to_drop if col in df[i].columns]
        
        if irrelevant_columns: 
            df[i].drop(columns=irrelevant_columns, axis=1, inplace=True)
            print("DataFrame", i , "after dropping irrelevant columns:")
            print(df[i].head())  # Printing the DataFrame after dropping columns
            print("-------------------------")
        else:
            print("DataFrame", i, "does not have any irrelevant columns")
            print("-------------------------")


  #[37]:


drop_irrelevant(df)


  #[38]:


batsman_df.head()


  #[39]:


encode_cols_batsman = ['Bat1','Runs','BF','SR','4s','6s','Opposition','Ground','Match_ID','Batsman']


  #[40]:


df_list=[batsman_df,bowler_df,ground_avg_df,match_results_df,match_total_df,players_df]


# ### Performing Feature Engineering (Encoding Categorical Columns)

  #[41]:


from sklearn.preprocessing import LabelEncoder


  #[42]:


# As we seen there are object type of data avilabe in the all the dataset so lets seaprate them and encode them into numeric form .
# lets seaprate the catogorical data & numerical data

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def get_categorical_columns(df_list):
    cat_col_list = []  # List to store categorical column names for each DataFrame

    for df in df_list:
        cat_col = []  # List to store categorical column names for the current DataFrame
        features = df.columns.values.tolist()

        for col in features:
            if df[col].dtype not in numerics:  # Check if the column's data type is non-numeric
                cat_col.append(col)

        cat_col_list.append(cat_col)

    return cat_col_list

def label_encode_categorical_columns(df_list):
    label = LabelEncoder()

    categorical_columns_list = get_categorical_columns(df_list)

    for i, cat_col in enumerate(categorical_columns_list):
        for col in cat_col:
            encoded_values = label.fit_transform(df_list[i][col])
            df_list[i][col] = encoded_values 


  #[43]:


label_encode_categorical_columns(df_list) 


  #[44]:


batsman_df.sample(10)


  #[45]:


match_results_df.sample(10)


  #[46]:


match_results_df.isnull().sum()


  #[47]:


match_total_df.isnull().sum()


  #[48]:


## performing the iterative imputation to columns match_results ,match_total
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
df_list2 =[match_results_df,match_total_df]

def treat_missing_values(df_list2):
    for i in range(len(df_list2)):
        df_imputed = imputer.fit_transform(df_list2[i])
        df_list2[i]=pd.DataFrame(df_imputed,columns =df_list2[i].columns)
        print("Data Frame",i,"after imputation")
        print(df_list2[i].isnull().sum().sum())


  #[49]:


treat_missing_values(df_list2)


  #[50]:


match_total_df = df_list2[0]

match_results_df = df_list2[1]


  #[51]:


match_total_df.isnull().sum()


  #[52]:


match_results_df.head()


  #[53]:


batsman_df.head()


  #[54]:


bowler_df.head()


  #[55]:


print(batsman_df.columns)
print('*' * 100)
print(bowler_df.columns)


# ### Performing Of Merging Operations (Inner) On all Sub-datasets based on common columns

  #[56]:


#Lets Create a master dataset and the build the model 
# so first we are going to merge 2 dataset and then further  
batsman_join_bowler=pd.merge(batsman_df,bowler_df,on=['Match_ID','Player_ID','Opposition','Ground','Month','Day','year'], how='inner')
batsman_join_bowler.shape




batsman_join_bowler.columns





#Lets Join ground_df
batsman_join_bowler_GrondAvg=pd.merge(batsman_join_bowler,ground_avg_df,on=['Ground'], how='inner')
batsman_join_bowler_GrondAvg




#Lets Join match_results & match_totals
OD_Total_result=pd.merge(match_results_df,match_total_df,on=['Ground','Country','Country_ID','Month','Day','year','Opposition'], how='inner')
OD_Total_result


#Lets Join batsman_join_bowler_GrondAvg & OD_Total_result
batsman_join_bowler_GrondAvg_OD=pd.merge(batsman_join_bowler_GrondAvg,OD_Total_result,on=['Ground','Month','Day','year'], how='inner')
batsman_join_bowler_GrondAvg_OD



players_df = players_df.rename(columns={'ID':'Player_ID'})



players_df.head()



master_df_after_join = pd.merge(batsman_join_bowler_GrondAvg_OD,players_df,on =['Player_ID','Country'],how='inner')



master_df_after_join['Batting Average'] = master_df_after_join['Bat1']/master_df_after_join['Inns']



master_df_after_join['Bowling Average'] = master_df_after_join['Runs_y'] / master_df_after_join['Wkts_y']



# filtering the rows that consist of zeros  in a way by considering only the list 
# of values into my master_df 

master_df_after_join = master_df_after_join[master_df_after_join['BF']>0]


master_df_after_join.head()


# ### Calculating strike rate 




# calculate strike rate of Batting for each player
master_df_after_join['strike Rate (Batting)'] = (master_df_after_join['Bat1']/master_df_after_join['BF'])*100


master_df_after_join.head()


master_df_after_join['Economy Rate (Bowling)'] = (master_df_after_join['Runs_y']/master_df_after_join['Overs_y'])


# Calculate the total Maiden Overs for each player
master_df_after_join['maiden Overs Total'] = master_df_after_join['Mdns'].sum()

master_df_after_join.head()


  #[73]:


batting_average = master_df_after_join['Batting Average']
bowling_average = master_df_after_join['Bowling Average']
strike_rate_batting = master_df_after_join['strike Rate (Batting)']
economy_rate_bowling = master_df_after_join['Economy Rate (Bowling)']
maiden_overs = master_df_after_join['maiden Overs Total']


# ### Normalizing The Data




def min_max_scaling(x):
    return(x - x.min()) / (x.max() - x.min())


#Step 2: Normalize the selected performance metrics
# You can use Min-Max Scaling or Z-score normalization
normalized_batting_average = min_max_scaling(batting_average)
normalized_bowling_average = min_max_scaling(bowling_average) 
normalized_strike_rate_batting = min_max_scaling(strike_rate_batting)
normalized_economy_rate_bowling = min_max_scaling(economy_rate_bowling)



# Step 3: Assign weights to each performance metric

batting_weight = 0.3 

bowling_weight = 0.25

strike_rate_weight = 0.2

economy_rate_weight = 0.25


# Step 4: Calculate the composite performance score for each player
master_df_after_join['Player Performance Score'] = (
    batting_weight * normalized_batting_average +
    bowling_weight * normalized_bowling_average +
    strike_rate_weight * normalized_strike_rate_batting +
    economy_rate_weight * normalized_economy_rate_bowling )


master_df_after_join



master_df_after_join.shape 


# ### Detecting and removing Outliers 


master_df_after_join.skew() 



master_df_after_join.skew()[(master_df_after_join.skew() >= 0.5) & (master_df_after_join.skew() >= -0.5)]


skew_col=['4s','6s','Ground','Player_ID','Overs_x','Mdns','Runs_y','Wkts_x','Econ','Ave_x','SR_y','Mat','Won','Runs', 'Wkts_y', 'Balls', 'Ave_y', 'RPO_x','BR','Country_ID','Bowling Average','Bowling Average','Economy Rate (Bowling)']
len(skew_col)

#Let's see the how data is distributed or Graphical analysis of all features
plt.figure(figsize=(15,20))
plotnumber=1
for column in master_df_after_join:
    if plotnumber<=60:
        ax=plt.subplot(10,6,plotnumber)
        sns.histplot(master_df_after_join[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.tight_layout()


# Now almost All skewness we removed so move further
#Let's check for the Outliers
plt.figure(figsize=(15,20))
plotnumber=1
for column in master_df_after_join:
    if plotnumber<=60:
        ax=plt.subplot(10,6,plotnumber)
        sns.boxplot(master_df_after_join[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.tight_layout()



# So from graph we can see the outlier are present in the some columns so lets remove them first 
outliers_col=['Margin', 'BR','Overs_y','Target','Ground','Span','strike Rate (Batting)','Player Performance Score']



master_df_after_join.head(5)


  #[87]:


# Remove the outliers by using Z score
from scipy.stats import zscore

z_score=zscore(master_df_after_join[outliers_col])
z_score_abs=np.abs(z_score)
filter_entry=(z_score_abs<3).all(axis=1)
master_after_join=master_df_after_join[filter_entry]
master_after_join.head()


  #[88]:


# Use 'Player Performance Score' as the target variable for the prediction model
x = master_after_join[['Bat1', 'Runs_x', 'BF', 'SR_x', '4s', '6s', 'Opposition_x', 'Ground',
          'Match_ID', 'Batsman', 'Player_ID', 'Month', 'Day', 'year', 'Overs_x',
          'Mdns', 'Runs_y', 'Wkts_x', 'Econ', 'Ave_x', 'SR_y', 'Bowler', 'Span',
          'Mat', 'Won', 'Tied', 'NR', 'Runs', 'Wkts_y', 'Balls', 'Ave_y', 'RPO_x',
          'Result_x', 'Margin', 'BR', 'Toss', 'Bat', 'Opposition_y', 'Match_ID_x',
          'Country', 'Country_ID', 'Score', 'Overs_y', 'RPO_y', 'Target', 'Inns',
          'Result_y', 'Match_ID_y']]


  #[89]:


y = master_after_join['Player Performance Score']  # Target column



print(x.shape, y.shape)


# Performing stadardization
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()
x_scaled=scale.fit_transform(x)


# ### splitting the data into training and testing 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


# ### Fitting the data into model and performing training 

lr = LinearRegression()
lr.fit(X_train,y_train)


# ### making some predictions


predictions = lr.predict(X_test)


# ### Perfroming model evaluation methods to see how well the model performance 


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r_squared = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nR-squared: {r_squared}")

