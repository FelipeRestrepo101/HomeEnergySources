from shiny import ui

def power_sources_ui():
    return ui.nav_panel("Power Sources",
        ui.input_text_area("textarea", "Enter Zip Code (85318)", ""),
        ui.input_date_range("date", "Choose Date (1/1/2025)"),
        ui.input_select("SourceChoice", "View Power Source trends", {
            "Solar": "Solar", "Wind": "Wind", "Coal": "Coal", "Hydro": "Hydro",
            "Petroleum": "Petroleum", "Nuclear": "Nuclear", "Natural Gas": "Natural Gas"
        }),
        ui.input_slider("Smooth", "Change rolling window (Smooth out lineplot)", min=1, max=365, value=1),
        ui.output_data_frame("zipcode_df"),
        ui.output_plot("sum_plot"),
        ui.output_plot("source_plot")
    )