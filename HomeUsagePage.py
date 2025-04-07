# BTU = 30
# with ui.nav_panel("Home Usage"):
#     ui.input_checkbox_group(
#         'Refrigerators',
#         'Refrigerators',
#         {1 : "Primary Fridge", 2 : "Second Fridge", 3: "Separate Freezers"}
#     )

#     HomeConsumption_df = pd.read_excel("data/ce5.3.st.xlsx", sheet_name="Btu", skiprows=3)
#     #Should add filter to automatically drop all states except users state. 
#     HomeConsumption_df.drop([1,62,63], inplace=True)
#     HomeConsumption_df.dropna(how='all', inplace=True)
#     HomeConsumption_df.set_index('Unnamed: 0', inplace=True)

#     LightbulbIcon = ui.HTML(
#         '''<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" fill="currentColor" class="bi bi-lightbulb" viewBox="0 0 16 16">
# <path d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13a.5.5 0 0 1 0 1 .5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1 0-1 .5.5 0 0 1 0-1 .5.5 0 0 1-.46-.302l-.761-1.77a2 2 0 0 0-.453-.618A5.98 5.98 0 0 1 2 6m6-5a5 5 0 0 0-3.479 8.592c.263.254.514.564.676.941L5.83 12h4.342l.632-1.467c.162-.377.413-.687.676-.941A5 5 0 0 0 8 1"/>
# </svg>'''
#     )

#     #pseudocode: depending on checkbox input, run if statement that determines how many of the three refrigeration columns to add. 


#     with ui.value_box(showcase=LightbulbIcon, theme="bg-gradient-indigo-purple"):
#         "Annual BTU usage"
#         f"{BTU} million"