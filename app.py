import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    with ui.layout_column_wrap():
        with ui.card():    
            ui.input_text_area("textarea", "Enter Zip Code (eg. 85302)", value="85302")
        with ui.card():
            ui.input_date_range("date", "Choose Date (eg. 1/1/2025)", start='2025-03-01', end='2025-04-01')

    iou = pd.read_csv("data/iou_zipcodes_2023.csv")
    noniou = pd.read_csv("data/non_iou_zipcodes_2023.csv")
    zip = pd.concat([noniou, iou], ignore_index=True)
    zip['zip'] = zip['zip'].astype(str)

    # EIA861 is the form from which the following excel file comes from, containing both 'eiaid' and 'BA ID' needed for merging
    EIA861 = pd.read_excel('data/Balancing_Authority_2023.xlsx')
    EIA861.rename(columns={'BA ID' : 'eiaid'}, inplace=True)
    EIA861 = EIA861[['eiaid', 'BA Code']]

    MergedZipcodes = pd.merge(zip, EIA861, on='eiaid', how='left')



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
        


    #Fetch data through API based on user specified date range
    @reactive.calc
    def fetch_power_source_data():
        try:
            url = "https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/"
            params = {
                "api_key": "jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs",
                "frequency": "daily",
                "data[0]": "value",
                "facets[timezone][]": "Arizona",
                "facets[respondent][]": get_URL_inputs()['BalancingAuthority'],
                "start": get_URL_inputs()['formatted_date1'],
                "end": get_URL_inputs()['formatted_date2'],
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": 0,
                "length": 5000
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get("response", {}).get("data", [])).drop(columns=['respondent-name'])  # Ensure safe extraction
            else:
                return pd.DataFrame({"Error": ["Failed to fetch data"]})  # Handle API failures
        
        except Exception as e:
            return pd.DataFrame({"Error": ["Failed to fetch data"]})  # Handle API failures



    @reactive.calc
    def fetch_demand_data():
        try:
            url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
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
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get("response", {}).get("data", []))
            else:
                return pd.DataFrame({"Error": ["Failed to fetch data"]})
        except Exception as e:
            return pd.DataFrame({"Error": ["Failed to fetch data"]})
    


    @reactive.calc
    def fetch_hourly_demand_data():
        try:
            url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
            params = {
                "api_key": "jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs",
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": f"{get_URL_inputs()['BalancingAuthority']}",
                "facets[type][]": "D",
                "start": f"{get_URL_inputs()['formatted_date1']}T00",
                "end": f"{get_URL_inputs()['formatted_date2']}T00",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": 0,
                "length": 5000
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get("response", {}).get("data", []))
            else:
                return pd.DataFrame({"Error": ["Failed to fetch data"]})
        except Exception as e:
            return pd.DataFrame({"Error": ["Failed to fetch data"]})

    with ui.card():
        #Zipcode_df
        @render.data_frame 
        def Zipcode_df():
            # Dynamically query the DataFrame
            QueryResult = zip.query(f"zip == '{input.textarea()}'")
            return render.DataGrid(QueryResult)



    #try except is necessary to avoid loading errors, making sure to wait until user has inputed date range first, and dataframe is created, before trying to
    # build the plot.
    with ui.card():
        with ui.layout_columns(min_height=800):
            # @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
            @render.plot()
            def power_sources_sum_plot():  
                try:
                    df = fetch_power_source_data()
                    #convert 'value' column to int because all columns are string 'object' type by default
                    df['value']  = df['value'].astype(int)

                    #groups by 'type-name' such as Coal, Natural Gas, Nuclear, etc. and then sums up all numeric values in each grouping, which in this case is just the value column
                    #loses all other column data because of .sum(numeric_only)
                    df = df.groupby('type-name').sum(numeric_only=True)#.to_frame()
                    df.index = df.index.str.replace('Solar with integrated battery storage', 'Solar with integrated\nbattery storage')

                    # plt.figure(figsize=(4,4))
                    graph = sns.barplot(df, x='type-name', y='value', palette='flare', hue='type-name') 
                    graph.set_title("Energy Production Totals")
                    graph.set_xlabel("Power Source")
                    graph.set_ylabel("MWh")
                    plt.gcf().axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) #puts commans in big numbers
                    # plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(True)
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
                
            @render.data_frame
            def power_sources_sum_grid():
                try:
                    df = fetch_power_source_data()
                    df['value']  = df['value'].astype(int)
                    df = df.groupby('type-name')[['value']].sum().reset_index()
                    df.rename(columns={'value': 'MegaWatt-hours'}, inplace=True)
                    df['MegaWatt-hours'] = df['MegaWatt-hours'].apply('{:,}'.format)
            
                    return df
                except Exception as e: 
                    return pd.DataFrame()  # Return empty DataFrame if no dates selected
                


        with ui.layout_columns(min_height=800):
            @render.plot()
            def power_sources_avg_plot():
                try:
                    df = fetch_power_source_data()
                    df['value'] = df['value'].astype(int)
                    df = df.groupby('type-name').mean(numeric_only=True)
                    df.index = df.index.str.replace('Solar with integrated battery storage', 'Solar with integrated\nbattery storage')

                    graph = sns.barplot(df, x='type-name', y='value', palette='flare', hue='type-name')
                    graph.set_title("Energy Production Daily Avg")
                    graph.set_xlabel("Power Source Avg") 
                    graph.set_ylabel("MWh")
                    plt.gcf().axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) #puts commans in big numbers
                    # plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
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


            @render.data_frame
            def power_sources_avg_grid():
                try:
                    df = fetch_power_source_data()
                    df['value']  = df['value'].astype(int)
                    df = df.groupby('type-name')[['value']].mean().reset_index()
                    df.rename(columns={'value': 'MegaWatt-hours'}, inplace=True)
                    df['MegaWatt-hours'] = df['MegaWatt-hours'].apply('{:,.0f}'.format) #puts commas and rounds decimal
                    return df
                except Exception as e: 
                    return pd.DataFrame()  # Return empty DataFrame if no dates selected

            

    with ui.card():
        with ui.layout_columns(col_widths=(5, 7)):
            with ui.layout_sidebar():
                with ui.sidebar(width=500, padding=50, bg='#f2f2f2'):               
                    #Individual Power source trends   
                    ui.input_select(  
                    "SourceChoice",  
                    "View Power Source Trends",  
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
            def single_power_source_plot():  
                try:
                    df = fetch_power_source_data()
                    df = df.query(f'`type-name` == "{input.SourceChoice()}"').copy()
                    df['value']  = df['value'].astype(int)
                    df['period'] = pd.to_datetime(df['period'])
                    df['value'] = df['value'].rolling(window=input.Smooth(), center=True, min_periods=1).mean()

                    graph = sns.lineplot(df, x='period', y='value')
                    graph.set_ylabel('Mwh')
                    plt.title(f"{input.SourceChoice()} Production Trends Over Time")
                    plt.xticks(rotation=90)
                    plt.gcf().axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) #puts commans in big numbers
                    # plt.tight_layout()
                    return graph
                
                except Exception as e:
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, 
                            "No data available\nPlease select valid dates", 
                            ha='center', 
                            va='center')
                    ax.set_axis_off()
                    return fig


      
    #Average Daily Demand by balancing authority
    with ui.card(): 
        @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
        def demand_plot():  
            try: 
                df = fetch_demand_data()
                df['value']  = df['value'].astype(int)
                df['period'] = pd.to_datetime(df['period'])

                graph = sns.lineplot(df, x='period', y='value')
                graph.set_ylabel('MWh')
                plt.title('Daily Energy Demand')
                plt.xticks(rotation=90) 
                plt.gcf().axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) #puts commans in big numbers
                return graph
            
            except Exception as e:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, 
                        "No data available\nPlease select valid dates", 
                        ha='center', 
                        va='center')
                ax.set_axis_off()
                return fig
            
    with ui.card(): 
        @render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
        def hourly_demand_plot():  
            try: 
                df = fetch_hourly_demand_data()
                df['value']  = df['value'].astype(int)
                df['period'] = pd.to_datetime(df['period'])
                df = df.groupby(df['period'].dt.hour)[['value']].mean()

                graph = sns.lineplot(df, x='period', y='value')
                graph.set_ylabel('MWh')
                plt.title('Hourly Energy Demand')
                plt.xticks(rotation=90) 
                plt.gcf().axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) #puts commans in big numbers
                return graph
            
            except Exception as e:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, 
                        "No data available\nPlease select valid dates", 
                        ha='center', 
                        va='center')
                ax.set_axis_off()
                return fig

    with ui.card():
        ui.card_header('Predictions')
    