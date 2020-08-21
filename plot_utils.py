import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from datetime import *
# from sklearn.metrics import *
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# global_mobility = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=40cc349b4e4ae7b5')

us_state_abbrev = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

unemployment_release_dates = {
    'Jan.' : '1/3/2020',
    'Feb.' : '2/7/2020',
    'March': '3/6/2020',
    'April': '4/3/2020',
    'May'  : '5/1/2020',
    'June' : '6/5/2020',
}

jan = pd.to_datetime('1/1/2020').to_period('M')
feb = pd.to_datetime('2/1/2020').to_period('M')
mar = pd.to_datetime('3/1/2020').to_period('M')
apr = pd.to_datetime('4/1/2020').to_period('M')
may = pd.to_datetime('5/1/2020').to_period('M')
jun = pd.to_datetime('6/1/2020').to_period('M')
jul = pd.to_datetime('7/1/2020').to_period('M')
aug = pd.to_datetime('8/1/2020').to_period('M')
sep = pd.to_datetime('9/1/2020').to_period('M')
oct = pd.to_datetime('10/1/2020').to_period('M')
nov = pd.to_datetime('11/1/2020').to_period('M')
dec = pd.to_datetime('12/1/2020').to_period('M')

date_to_month = {
    jan : 'Jan',
    feb : 'Feb',
    mar : 'Mar',
    apr : 'Apr',
    may : 'May',
    jun : 'Jun',
    jul : 'Jul',
    aug : 'Aug',
    sep : 'Sep',
    oct : 'Oct',
    nov : 'Nov',
    dec : 'Dec'
}

def is_increasing(change):
    if change < 0:
        return 0
    #elif change > 0:
        #return 1
    else:
        return 1

def make_categorical(df):
    df['Unemployment Rate Change'] = df['Unemployment Rate Change'].map(is_increasing)

def normalize_by_pop(df,scale = 100000):
    X = df.drop(columns = ['date','state','Unemployment Rate (S)','Unemployment Rate Change'])
    cols = X.columns
    count_row = df.shape[0]

    for col in cols:
        for i in range(count_row):
            df.iloc[col][i] = (df.iloc[col][i]/df['Population']) * scale

def log_reg_summary(y_true,y_pred):
    print("Accuracy score is : " + str(accuracy_score(y_true,y_pred)))
    print("Precision score is : " + str(precision_score(y_true, y_pred)))
    print("Recall score is : " + str(recall_score(y_true,y_pred)))
    print("F1 score is : " + str(f1_score(y_true,y_pred)))
    print("ROC AUC is : " + str(roc_auc_score(y_true,y_pred)))

# https://stackoverflow.com/questions/51023806/how-to-get-adjusted-r-square-for-linear-regression
# adj_r2 ^^

def linear_reg_summary(data, y_true,y_pred):
    r2 = r2_score(y_true,y_pred)
    
    n = data.count()[0]
    k = len(data.columns)
    
    adj_r2 = (1 - (1 - r2) * ((n - 1) / 
             (n - k - 1)))
    
    print("R^2 is :                   " + str(r2))
    print("Adjusted R2 is :           " + str(adj_r2))
    print("MSE is :                   " + str(mean_squared_error(y_true,y_pred)))
    print("Mean absolute error is :   " + str(mean_absolute_error(y_true,y_pred)))
    print("Median absolute error is : " + str(median_absolute_error(y_true,y_pred)))
    # print("Mean squared error is : " + str(mean_squared_error(y_true,y_pred)))

    # "R^2: {:.2f}".format(r2)



def month_to_str(m):
    key = m.to_period('M')
    return key#date_to_month[key]

def dates_to_month_str(df):
    df['date'] = df['date'].map(month_to_str)


def str_to_date(s):
    return pd.to_datetime(s)


def num_to_date(num):
    date = num % 100
    num = num // 100
    month = num % 100
    year = num // 100
    return pd.to_datetime((str(month) + "/" + str(date) + "/" + str(year)))

def fix_date_type(df):
    df['date'] = df['date'].map(num_to_date)


def code_to_state(code):
    return us_state_abbrev[code]

def fix_state_code(df):
    df['state'] = df['state'].map(code_to_state)

def month_to_date(month):
    return pd.to_datetime(unemployment_release_dates[month])

def fix_month(df):
    df['date'] = df['date'].map(month_to_date)


# BELOW ARE THE OLD GRAPHING FUNCTIONS FOR JUST COVIDTRACKING.COM


def line_graph(df, states, x = 15, y = 10,font = 25, alpha_val = 0.7, legend_font = 10, rotate = True,
               y_axis = "positiveIncrease", log_y = False, rolling_avg = False, rolling_avg_len = 7,
               filename = '2019_US_Pop.csv'):
    
    census = pd.read_csv(filename)
    fig, ax = plt.subplots(figsize = (x, y))
    
    for state in states:
        state_df = df[df['state'] == state]

        if(log_y):
            plt.plot(state_df['date'], np.log(state_df[y_axis]), alpha = alpha_val, label = "Log of " + y_axis + " in " + state)
        else:
            plt.plot(state_df['date'], state_df[y_axis], alpha = alpha_val, label = y_axis + " in " + state)

        if(rolling_avg):
            pd.options.mode.chained_assignment = None
            new_col_name = str(rolling_avg_len) + 'ra_' + str(y_axis)
            state_df[new_col_name] = state_df[y_axis].rolling(rolling_avg_len).mean()
            plt.plot(state_df['date'], state_df[new_col_name], alpha = alpha_val, label = str(rolling_avg_len) + ' day rolling average for ' + str(state))
        
    
    if(log_y):
        plt.title("Log of " + y_axis + " vs Date", fontsize = font)
        plt.ylabel("Log of " + y_axis, fontsize = font)
    else:
        plt.title(y_axis + " vs Date", fontsize = font)
        plt.ylabel(y_axis, fontsize = font)
        
    plt.legend(fontsize = legend_font)
    plt.xlabel("Date", fontsize = font)
        

def box_plot(df, states, census_data = "2019_US_Pop.csv", x = 15, y = 10,font = 25, alpha_val = 0.7, legend_font = 10, rotate = True,
               y_axis = "positiveIncrease", log_y = False, norm_x = False):

    state_df = df[covid.state.isin(states)]
    
    state_df.boxplot(by = "state", column=[y_axis], figsize = (x,y), fontsize = font)
    
    plt.title("Boxplot of " + y_axis + " by State")
    plt.ylabel(y_axis, fontsize = font)
    plt.xlabel("State", fontsize = font)

def top_n(df, n = 5, measure = 'positiveIncrease', date = date.today() - timedelta(days=1), top = True):
    date_filter = df[df['date'] == pd.to_datetime(date)]

    date_filter = date_filter[['date', 'state', measure]]
    
    if (top):
        return date_filter.nlargest(n, measure)
    else:
        return date_filter.nsmallest(n, measure)

def scatter_plot(df, states, x_axis = 'deathIncrease', y_axis = 'positiveIncrease', s = None, c = None, font = 25, legend_font = 10):
    # df.plot.scatter(x_axis, y_axis)
    for state in states:
        state_df = df[df['state'] == state]
        plt.scatter(state_df[[x_axis]], state_df[[y_axis]], label = state)

    positivity_rates = [0.05, 0.1, 0.2]

    for rate in positivity_rates:
        temp = df[df.state.isin(states)]
        nmax = temp['negative'].max()
        plt.plot([0,nmax], [0, nmax*rate/(1-rate)], 'go--', alpha = 0.5, label = str(rate*100) + "% positivity rate")

    plt.title("Scatterplot of " + y_axis + " against " + x_axis)
    plt.ylabel(y_axis, fontsize = font)
    plt.xlabel(x_axis, fontsize = font)

    temp = df[df.state.isin(states)]
    pmax = temp['positive'].max()

    plt.ylim(0,pmax)

    plt.legend(fontsize = legend_font)
