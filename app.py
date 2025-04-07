from shiny import App, reactive, render, ui
from shiny_validate import InputValidator, check
from modules.data_processing import load_zipcode_data, load_balancing_authority_data, merge_zip_and_authority_data, fetch_api_data
from modules.plots import plot_energy_totals, plot_source_trend
from modules.ui_components import power_sources_ui

app_ui = ui.page_navbar(
    power_sources_ui(),
    title="YourPowerGrid",
    bg='#ff6600'
)

def server(input, output, session):
    iv = InputValidator()
    iv.add_rule("textarea", check.required())
    iv.add_rule("date", check.required())
    iv.enable()

    zip_data = load_zipcode_data()
    eia861 = load_balancing_authority_data()
    merged_data = merge_zip_and_authority_data(zip_data, eia861)

    @reactive.calc
    def get_balancing_authority():
        req(iv.is_valid())
        zip_code = input.textarea().strip()
        result = merged_data.query(f"zip == '{zip_code}'")
        if not result.empty:
            return result['BA Code'].iat[0]
        return None

@reactive.calc
def get_balancing_author():
    zip_code = input.zip_code()
    if zip_code in zip_to_authority["zip"].values:
        return zip_to_authority[zip_to_authority["zip"] == zip_code]["authority"].values[0]
    else:
        return None

    @reactive.calc
    def get_api_data():
        balancing_authority = get_balancing_author()
        if not balancing_authority:
            return pd.DataFrame()
        return fetch_api_data(balancing_authority, str(input.date()[0]), str(input.date()[1]))






        
    
    