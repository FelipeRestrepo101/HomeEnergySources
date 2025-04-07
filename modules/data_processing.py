import pandas as pd
import requests

def load_zipcode_data():
    iou = pd.read_csv("data/iou_zipcodes_2023.csv")
    noniou = pd.read_csv("data/non_iou_zipcodes_2023.csv")
    zip_data = pd.concat([noniou, iou], ignore_index=True)
    zip_data['zip'] = zip_data['zip'].astype(str)
    return zip_data

def load_balancing_authority_data():
    eia861 = pd.read_excel('data/Balancing_Authority_2023.xlsx')
    eia861.rename(columns={'BA ID': 'eiaid'}, inplace=True)
    return eia861[['eiaid', 'BA Code']]

def merge_zip_and_authority_data(zip_data, eia861):
    return pd.merge(zip_data, eia861, on='eiaid', how='left')

def fetch_api_data(balancing_authority, start_date, end_date):
    url = f"https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/"
    params = {
        "api_key": "your_api_key",
        "frequency": "daily",
        "data[0]": "value",
        "facets[timezone][]": "Arizona",
        "facets[respondent][]": balancing_authority,
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data.get("response", {}).get("data", [])).drop(columns=['respondent-name'])
    else:
        return pd.DataFrame({"Error": ["Failed to fetch data"]})