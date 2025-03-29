import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from shiny.express import ui, input, render
from shiny import reactive


# data link
#https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48 

with ui.sidebar():
    ui.input_text_area("textarea", "Enter Zip Code (85318)", "")
    #used to inspect input component result.
    # @render.text
    # def textbox():
    #     return input.textarea()

    ui.input_date_range("date", "Choose Date (1/1/2025)")
    # used to inspect input component result.
    # @render.text
    # def datebox():
    #     # Cross-platform approach to format date without leading zeros
    #     selected1 = input.date()[0]
    #     selected2 = input.date()[1]
    #     formatted_date1 = selected1.strftime("%Y-%m-%d")
    #     formatted_date2 = selected2.strftime("%Y-%m-%d")
    #     return formatted_date1, formatted_date2

iou = pd.read_csv("iou_zipcodes_2023.csv")
noniou = pd.read_csv("non_iou_zipcodes_2023.csv")
zip = pd.concat([noniou, iou], ignore_index=True)
zip['zip'] = zip['zip'].astype(str)
 

@render.data_frame
def zipcode_df():
    # Dynamically query the DataFrame
    result = zip.query(f"zip == '{input.textarea()}'")
    return render.DataGrid(result)


#Fetch data through API based on user specified date range
@reactive.calc
def fetch_api_data():
    selected1 = input.date()[0]
    selected2 = input.date()[1]
    formatted_date1 = selected1.strftime("%Y-%m-%d")
    formatted_date2 = selected2.strftime("%Y-%m-%d")


#     url = "https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/?\
# api_key=jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs&frequency=daily&data[0]=value&facets[timezone][]=Arizona&facets[respondent][]=AZPS&\
# start=2025-01-01&end=2025-01-31&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"

    url = f"https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/?\
api_key=jKuhIenGf4YPfA88Y1VvFTLTBcXo6gYVCUOnNoFs&frequency=daily&data[0]=value&facets[timezone][]=Arizona&facets[respondent][]=AZPS&\
start={formatted_date1}&end={formatted_date2}&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"

    response = requests.get(url)

    # #no error handling
    # data = response.json()
    # return pd.DataFrame(data['response']['data'])

    # error handling alternative
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data.get("response", {}).get("data", []))  # Ensure safe extraction
    else:
        return pd.DataFrame({"Error": ["Failed to fetch data"]})  # Handle API failures


#make datresult reactive in shiny framework so it is accesible between functions
dateresult = reactive.value(None)



@render.data_frame
def date_result():
    if input.date() is not None and len(input.date()) == 2:
        dateresult.set(fetch_api_data())
        return dateresult.get()
    else: 
        return pd.DateFrame()  # Return empty DataFrame if no dates selected


# @reactive.Effect
# def info():
#     dateresult.get().info()

@render.plot(width= 800, height=800, alt="A Seaborn histogram on penguin body mass in grams.")  
def sum_plot():  

    
    if (input.date() is not None and len(input.date()) == 2 and 
        dateresult() is not None and not dateresult().empty):
        dateframe = dateresult.get()
        # dateframe['TotalEnergy'] = dateframe.filter(like='MWh').sum(axis=1)
        # sum = dateframe.sum(numeric_only=True).to_frame()
        # sum.reset_index(inplace=True)

        dateframe['value']  = dateframe['value'].astype(int)

        #loses all other column data because of .sum(numeric_only)
        sum = dateframe.groupby('type-name').sum(numeric_only=True)#.to_frame()

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