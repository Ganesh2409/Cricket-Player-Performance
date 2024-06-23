# Cricket Performance Analysis

## Overview

This project involves analyzing cricket player performance using various datasets related to batting, bowling, match results, and player statistics. The goal is to perform data preprocessing, feature engineering, and model training to predict and evaluate player performance scores.

## Project Structure

The project is structured as follows:

```
Cricket-Player-Performance/
│
├── Data/
│   ├── Batsman_Data.csv
│   ├── Bowler_data.csv
│   ├── Ground_Averages.csv
│   ├── ODI_Match_Results.csv
│   ├── ODI_Match_Totals.csv
│   └── WC_players.csv
│
├── README.md
├── requirements.txt
└── performance_analysis.py
```

- **Data/**: Directory containing CSV files for different datasets used in the analysis.
- **README.md**: This file, containing project overview, setup instructions, and usage details.
- **requirements.txt**: File listing dependencies required to run the project.
- **performance_analysis.py**: Python script with code for data analysis, preprocessing, modeling, and evaluation.

## Setup Instructions

### Requirements

Ensure you have Python 3.x installed along with necessary libraries listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ganesh2409/Cricket_Player_Performance_Prediction.git
   cd Cricket-Player-Performance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

To run the performance analysis script:

```bash
python performance_analysis.py
```

## Project Details

### Data Loading and Initial Exploration

- The script starts by loading datasets (`Batsman_Data.csv`, `Bowler_data.csv`, etc.) using pandas for initial exploration.

### Data Cleaning and Preprocessing

- **Handling Null Values**: Checks for null values in each dataset and performs necessary operations.
- **Date Separation**: Splits the 'Start Date' into day, month, and year columns across datasets.

### Feature Engineering

- **Dropping Irrelevant Columns**: Removes unnecessary columns ('Unnamed: 0', 'Start Date', 'Year') from datasets.
- **Encoding Categorical Columns**: Converts categorical data into numeric form using Label Encoding.

### Data Imputation

- **Iterative Imputation**: Uses iterative imputer to handle missing values in `match_results_df` and `match_total_df`.

### Data Merging

- **Inner Joins**: Merges datasets (`batsman_df`, `bowler_df`, `ground_avg_df`, `match_results_df`, `match_total_df`, `players_df`) on common columns to create a master dataset (`master_df_after_join`).

### Data Analysis and Visualization

- **Statistical Analysis**: Calculates batting average, bowling average, strike rate, economy rate, and more.
- **Data Normalization**: Applies Min-Max Scaling to normalize selected performance metrics.
- **Outlier Detection**: Identifies and removes outliers using Z-score method.

### Model Training and Evaluation

- **Linear Regression Model**: Splits data into training and testing sets, scales features, trains a Linear Regression model, and evaluates its performance using metrics like MAE, MSE, RMSE, and R-squared.

### Results
* Mean Absolute Error: 0.026260716919972147
* Mean Squared Error: 0.0010815997971432808
* Root Mean Squared Error: 0.032887684581667964
* R-squared: 0.9402293334532789

## Conclusion

This project provides insights into cricket player performance using data analysis and machine learning techniques. It aims to help cricket analysts and enthusiasts understand the factors influencing player performance scores.


![Logo]()

