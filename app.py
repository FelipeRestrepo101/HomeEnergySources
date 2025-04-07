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



ui.page_opts(
    title="YourPowerGrid",  
    page_fn=partial(page_navbar, id="page", bg='#ff6600'),  
)

with ui.nav_panel("Power Sources"):

    
# with ui.sidebar():
    ui.input_text_area("textarea", "Enter Zip Code (85318)", "")
    # # used to inspect input component result.
    # @render.text
    # def textbox():
    #     return input.textarea()

    ui.input_date_range("date", "Choose Date (1/1/2025)")
    # # used to inspect input component result.
    # @render.text
    # def datebox():
    #     # Cross-platform approach to format date without leading zeros
    #     selected1 = input.date()[0]
    #     selected2 = input.date()[1]
    #     formatted_date1 = selected1.strftime("%Y-%m-%d")
    #     formatted_date2 = selected2.strftime("%Y-%m-%d")
    #     return formatted_date1, formatted_date2

    iou = pd.read_csv("data/iou_zipcodes_2023.csv")
    noniou = pd.read_csv("data/non_iou_zipcodes_2023.csv")
    zip = pd.concat([noniou, iou], ignore_index=True)
    zip['zip'] = zip['zip'].astype(str)

    # EIA861 is the form from which the following excel file comes from, containing both 'eiaid' and 'BA ID' needed for merging
    EIA861 = pd.read_excel('data/Balancing_Authority_2023.xlsx')
    EIA861.rename(columns={'BA ID' : 'eiaid'}, inplace=True)
    EIA861 = EIA861[['eiaid', 'BA Code']]

    MergedZipcodes = pd.merge(zip, EIA861, on='eiaid', how='left')

    @render.data_frame #Zipcode_df
    def Zipcode_df():
        # Dynamically query the DataFrame
        QueryResult = zip.query(f"zip == '{input.textarea()}'")
        return render.DataGrid(QueryResult)


    #Fetch data through API based on user specified date range
    @reactive.calc
    def fetch_api_data():
        #if statement is needed for zip code input validation
        if len(input.textarea()) > 4:
            BalancingAuthority = MergedZipcodes.query(f"zip == '{input.textarea().strip()}'")['BA Code'].iat[0]
            selected1 = input.date()[0]
            selected2 = input.date()[1]
            formatted_date1 = selected1.strftime("%Y-%m-%d")
            formatted_date2 = selected2.strftime("%Y-%m-%d")

            #Dynamic URL (balancing authority and date range are dynamic)
            url = f"https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/?\
api_key=jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs&frequency=daily&data[0]=value&facets[timezone][]=Arizona&facets[respondent][]={BalancingAuthority}&\
start={formatted_date1}&end={formatted_date2}&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get("response", {}).get("data", [])).drop(columns=['respondent-name'])  # Ensure safe extraction
            else:
                return pd.DataFrame({"Error": ["Failed to fetch data"]})  # Handle API failures


    #make datresult reactive in shiny framework so it is accesible between functions
    PowerSource_df = reactive.value(None)


    @render.data_frame
    def PowerSource_df_func():
        if input.date() is not None and len(input.date()) == 2:
            PowerSource_df.set(fetch_api_data()) #attached df to reactive PowerSource_df above, otherwise PowerSource_df_func().get() would have to be used, PowerSource_df_func().get() should
            #still work if you wanted because it returns the reactive PowerSource_df in itself, but rather redundant for processing. 
            return PowerSource_df.get()
        else: 
            return pd.DataFrame()  # Return empty DataFrame if no dates selected



    #if else is necessary to avoid loading errors, making sure to wait until user has inputed date range first, and dataframe is created, before trying to
    # build the plot.
    @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
    def sum_plot():  
        if (input.date() is not None and len(input.date()) == 2 and 
            PowerSource_df.get() is not None and not PowerSource_df.get().empty):
            PS_df = PowerSource_df.get()

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
        
        else: 
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
        if (input.date() is not None and len(input.date()) == 2 and 
            PowerSource_df.get() is not None and not PowerSource_df.get().empty):
            PS_df = PowerSource_df.get()
            PS_df = PS_df.query(f'`type-name` == "{input.SourceChoice()}"').copy()
            PS_df['value']  = PS_df['value'].astype(int)
            PS_df['period'] = pd.to_datetime(PS_df['period'])
            PS_df['value'] = PS_df['value'].rolling(window=input.Smooth(), center=True, min_periods=1).mean()

            graph = sns.lineplot(PS_df, x='period', y='value') 
            plt.xticks(rotation=90)
            return graph
        
    @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
    def demand_plot():  
    #Balancing Authority Demand
        BalancingAuthority = MergedZipcodes.query(f"zip == '{input.textarea().strip()}'")['BA Code'].iat[0]
        selected1 = input.date()[0]
        selected2 = input.date()[1]
        formatted_date1 = selected1.strftime("%Y-%m-%d")
        formatted_date2 = selected2.strftime("%Y-%m-%d")

        url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        # Parameters (including API key)
        params = {
            "api_key": "jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs",
            "frequency": "daily",
            "data[0]": "value",
            "facets[respondent][]": f"{BalancingAuthority}",
            "facets[timezone][]": "Arizona",
            "facets[type][]": "D",
            "start": f"{formatted_date1}",
            "end": f"{formatted_date2}",
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




BTU = 30
with ui.nav_panel("Home Usage"):
    ui.input_checkbox_group(
        'Refrigerators',
        'Refrigerators',
        {1 : "Primary Fridge", 2 : "Second Fridge", 3: "Separate Freezers"}
    )

    HomeConsumption_df = pd.read_excel("data/ce5.3.st.xlsx", sheet_name="Btu", skiprows=3)
    #Should add filter to automatically drop all states except users state. 
    HomeConsumption_df.drop([1,62,63], inplace=True)
    HomeConsumption_df.dropna(how='all', inplace=True)
    HomeConsumption_df.set_index('Unnamed: 0', inplace=True)

    LightbulbIcon = ui.HTML(
        '''<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" fill="currentColor" class="bi bi-lightbulb" viewBox="0 0 16 16">
<path d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13a.5.5 0 0 1 0 1 .5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1 0-1 .5.5 0 0 1 0-1 .5.5 0 0 1-.46-.302l-.761-1.77a2 2 0 0 0-.453-.618A5.98 5.98 0 0 1 2 6m6-5a5 5 0 0 0-3.479 8.592c.263.254.514.564.676.941L5.83 12h4.342l.632-1.467c.162-.377.413-.687.676-.941A5 5 0 0 0 8 1"/>
</svg>'''
    )

    #pseudocode: depending on checkbox input, run if statement that determines how many of the three refrigeration columns to add. 


    with ui.value_box(showcase=LightbulbIcon, theme="bg-gradient-indigo-purple"):
        "Annual BTU usage"
        f"{BTU} million"


        
    
    