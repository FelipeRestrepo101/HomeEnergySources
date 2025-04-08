import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from functools import partial
from shiny.express import ui, input, render
from shiny.ui import page_navbar
from shiny import reactive

#used for navbar
ui.page_opts(
    title="YourPowerGrid",  
    page_fn=partial(page_navbar, id="page", bg='#ff6600'),  
)

with ui.nav_panel("Power Sources"):

    ui.input_text_area("textarea", "Enter Zip Code (85318)", "")
    ui.input_date_range("date", "Choose Date (1/1/2025)")


    iou = pd.read_csv("data/iou_zipcodes_2023.csv")
    noniou = pd.read_csv("data/non_iou_zipcodes_2023.csv")
    zip = pd.concat([noniou, iou], ignore_index=True)
    zip['zip'] = zip['zip'].astype(str)

    # EIA861 is the form from which the following excel file comes from, containing both 'eiaid' and 'BA ID' needed for merging
    EIA861 = pd.read_excel('data/Balancing_Authority_2023.xlsx')
    EIA861.rename(columns={'BA ID' : 'eiaid'}, inplace=True)
    EIA861 = EIA861[['eiaid', 'BA Code']]

    MergedZipcodes = pd.merge(zip, EIA861, on='eiaid', how='left')



    
    # Alternative 1: place variables in global environment:
    # BalancingAuthority = MergedZipcodes.query(f"zip == '{input.textarea().strip()}'")['BA Code'].iat[0]
    # selected1 = input.date()[0]
    # selected2 = input.date()[1]
    # formatted_date1 = selected1.strftime("%Y-%m-%d")
    # formatted_date2 = selected2.strftime("%Y-%m-%d")

    # Issue: variables are not reactive

    
    # Alternative 2: place variable in reactive components:
    # @reactive.calc
    # def get_URL_inputs():
    # BalancingAuthority = MergedZipcodes.query(f"zip == '{input.textarea().strip()}'")['BA Code'].iat[0]
    # selected1 = input.date()[0]
    # selected2 = input.date()[1]
    # formatted_date1 = selected1.strftime("%Y-%m-%d")
    # formatted_date2 = selected2.strftime("%Y-%m-%d")

    # Issue: variable are not global, and function can only return one value. Also 
    # reactive components are executed every time a dependency within changes, however doesnt account for invalid input such as same start and end date leading to error.
    

    #Solution: Return dictionary and add try or if blocks to attempt and execute reactive components, and return None if failed, then use try and if block in function accessing 
    # component to check if component returns None or not.
    @reactive.calc
    def get_URL_inputs():
        try: 
            BalancingAuthority = MergedZipcodes.query(f"zip == '{input.textarea().strip()}'")['BA Code'].iat[0]
            selected1 = input.date()[0]
            selected2 = input.date()[1]
            formatted_date1 = selected1.strftime("%Y-%m-%d")
            formatted_date2 = selected2.strftime("%Y-%m-%d")

            return {
                "BalancingAuthority": BalancingAuthority,
                "formatted_date1": formatted_date1,
                "formatted_date2": formatted_date2
            }

        except Exception as e:
            return None


    @render.data_frame #Zipcode_df
    def Zipcode_df():
        # Dynamically query the DataFrame
        QueryResult = zip.query(f"zip == '{input.textarea()}'")
        return render.DataGrid(QueryResult)


    #Fetch data through API based on user specified date range
    @reactive.calc
    def fetch_api_data():
        try:
            #Dynamic URL (balancing authority and date range are dynamic)
            url = f"https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/?\
api_key=jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs&frequency=daily&data[0]=value&facets[timezone][]=Arizona&facets[respondent][]={get_URL_inputs()['BalancingAuthority']}&\
start={get_URL_inputs()['formatted_date1']}&end={get_URL_inputs()['formatted_date2']}&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get("response", {}).get("data", [])).drop(columns=['respondent-name'])  # Ensure safe extraction
            else:
                return pd.DataFrame({"Error": ["Failed to fetch data"]})  # Handle API failures
        
        except Exception as e:
            return pd.DataFrame({"Error": ["Failed to fetch data"]})  # Handle API failures
        


    #if else is necessary to avoid loading errors, making sure to wait until user has inputed date range first, and dataframe is created, before trying to
    # build the plot.
    @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
    def sum_plot():  
        try:
            PS_df = fetch_api_data()

            #creates sum PowerSource_df containing a column that is a sum/total of all energy sources, previously used MWh filter which needs to be changed
            # dateframe['TotalEnergy'] = dateframe.filter(like='MWh').sum(axis=1)
            # sum = dateframe.sum(numeric_only=True).to_frame()
            # sum.reset_index(inplace=True)


            #convert 'value' column to int because all columns are string 'object' type by default
            PS_df['value']  = PS_df['value'].astype(int)

            #groups by 'type-name' such as Coal, Natural Gas, Nuclear, etc. and then sums up all numeric values in each grouping, which in this case is just the value column
            #loses all other column data because of .sum(numeric_only)
            sum = PS_df.groupby('type-name').sum(numeric_only=True)#.to_frame()

            # plt.figure(figsize=(4,4))
            graph = sns.barplot(sum, x='type-name', y='value', palette='flare', hue='type-name') 

            graph.set_title("Energy Totals")
            graph.set_xlabel("Power Source")
            graph.set_ylabel("Count")

            # graph.xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks on the x-axis
            # graph.yaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks on the y-axis

            plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
            plt.xticks(rotation=90)
            return graph
        
        except Exception as e: 
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 
                    "No data available\nPlease select valid dates", 
                    ha='center', 
                    va='center')
            ax.set_axis_off()
            return fig
        

    #Individual Power source trends   
    ui.input_select(  
    "SourceChoice",  
    "View Power Source trends",  
    {"Solar": "Solar", "Wind": "Wind", "Coal": "Coal", "Hydro": "Hydro", 
     "Petroleum": "Petroleum", "Nuclear": "Nuclear", "Natural Gas": "Natural Gas" },  
    )  

    ui.input_slider(
     "Smooth",
     "Change rolling window (Smooth out lineplot)",
     min=1,
     max=365,
     value=1
    )

    @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
    def source_plot():  
        try:
            PS_df = fetch_api_data()
            PS_df = PS_df.query(f'`type-name` == "{input.SourceChoice()}"').copy()
            PS_df['value']  = PS_df['value'].astype(int)
            PS_df['period'] = pd.to_datetime(PS_df['period'])
            PS_df['value'] = PS_df['value'].rolling(window=input.Smooth(), center=True, min_periods=1).mean()

            graph = sns.lineplot(PS_df, x='period', y='value') 
            plt.xticks(rotation=90)
            return graph
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 
                    "No data available\nPlease select valid dates", 
                    ha='center', 
                    va='center')
            ax.set_axis_off()
            return fig

      
    #Balancing Authority Demand  
    @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
    def demand_plot():  
        try: 

            url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
            # Parameters (including API key)
            params = {
                "api_key": "jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs",
                "frequency": "daily",
                "data[0]": "value",
                "facets[respondent][]": f"{get_URL_inputs()['BalancingAuthority']}",
                "facets[timezone][]": "Arizona",
                "facets[type][]": "D",
                "start": f"{get_URL_inputs()['formatted_date1']}",
                "end": f"{get_URL_inputs()['formatted_date2']}",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": 0,
                "length": 5000
            }

            response = requests.get(url, params=params)
            data = response.json()
            df = pd.DataFrame(data['response']['data'])
            df['value']  = df['value'].astype(int)
            df['period'] = pd.to_datetime(df['period'])

            graph = sns.lineplot(df, x='period', y='value')
            plt.title('Demand')
            plt.xticks(rotation=90) 


            return graph
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 
                    "No data available\nPlease select valid dates", 
                    ha='center', 
                    va='center')
            ax.set_axis_off()
            return fig






        
    
    