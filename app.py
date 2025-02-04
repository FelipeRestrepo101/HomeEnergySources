import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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



    ui.input_date_range("date", "Choose Date (1/1/2025)", format='mm-dd-yyyy')
    #used to inspect input component result.
    # @render.text
    # def datebox():
        # Cross-platform approach to format date without leading zeros
        # selected1 = input.date()[0]
        # selected2 = input.date()[1]
        # formatted_date1 = f"{selected1.month}-{selected1.day}-{selected1.year}"
        # formatted_date2 = f"{selected2.month}-{selected2.day}-{selected2.year}"        
        # return formatted_date1, formatted_date2



iou = pd.read_csv("iou_zipcodes_2023.csv")
noniou = pd.read_csv("non_iou_zipcodes_2023.csv")
zip = pd.concat([noniou, iou], ignore_index=True)
zip['zip'] = zip['zip'].astype(str)
 
resources = pd.read_csv("930-data-export (2).csv")
resources['Timestamp (Hour Ending)'] = resources['Timestamp (Hour Ending)'].str.replace('/', '-').str.replace(', Mountain Time', '')

#changes it to datetime format, otherwise query works incorrectly
resources['Timestamp (Hour Ending)'] = pd.to_datetime(resources['Timestamp (Hour Ending)'], format='%m-%d-%Y')

@render.data_frame
def result_df():
    # Dynamically query the DataFrame
    result = zip.query(f"zip == '{input.textarea()}'")
    return render.DataGrid(result)

#make datresult reactive in shiny framework so it is accesible between functions
dateresult = reactive.value(None)

@render.data_frame
def date_result():
    selected1 = input.date()[0]
    selected2 = input.date()[1]
    formatted_date1 = f"{selected1.month}-{selected1.day}-{selected1.year}"
    formatted_date2 = f"{selected2.month}-{selected2.day}-{selected2.year}"        
    dateresult.set(resources.query(f"`Timestamp (Hour Ending)` >= '{formatted_date1}' and `Timestamp (Hour Ending)` <= '{formatted_date2}'").copy())
    return dateresult.get()



@render.plot(width= 800, height=1000, alt="A Seaborn histogram on penguin body mass in grams.")  
def plot():  
    dateframe = dateresult.get()
    dateframe['TotalEnergy'] = dateframe.filter(like='MWh').sum(axis=1)
    sum = dateframe.sum(numeric_only=True).to_frame()
    sum.reset_index(inplace=True)


    graph = sns.barplot(sum, x='index', y=0, hue='index', palette='flare') 

    graph.set_title("Energy Totals")
    graph.set_xlabel("Power Source")
    graph.set_ylabel("Count")

    # graph.xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks on the x-axis
    # graph.yaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks on the y-axis

    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    plt.xticks(rotation=90)
    
    return graph
