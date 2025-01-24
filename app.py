import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny.express import ui, input, render

with ui.sidebar():
    ui.input_text_area("textarea", "Enter Zip Code (85318)", "")
    #used to inspect input component result.
    # @render.text
    # def textbox():
    #     return input.textarea()


    ui.input_date("date", "Choose Date (1/1/2025)", format='mm-dd-yyyy')
    #used to inspect input component result.
    # @render.text
    # def datebox():
    #     # Cross-platform approach to format date without leading zeros
    #     selected = input.date()
    #     formatted_date = f"{selected.month}-{selected.day}-{selected.year}"
    #     return formatted_date



iou = pd.read_csv("iou_zipcodes_2023.csv")
noniou = pd.read_csv("non_iou_zipcodes_2023.csv")
zip = pd.concat([noniou, iou], ignore_index=True)
zip['zip'] = zip['zip'].astype(str)
 
resources = pd.read_csv("930-data-export (2).csv")
resources['Timestamp (Hour Ending)'] = resources['Timestamp (Hour Ending)'].str.replace('/', '-').str.replace(', Mountain Time', '')

@render.data_frame
def result_df():
    # Dynamically query the DataFrame
    result = zip.query(f"zip == '{input.textarea()}'")
    return render.DataGrid(result)




@render.data_frame
def date_result():
    selected = input.date()
    formatted_date = f"{selected.month}-{selected.day}-{selected.year}"
    dateresult = resources.query(f"`Timestamp (Hour Ending)` == '{formatted_date}'")
    return dateresult


