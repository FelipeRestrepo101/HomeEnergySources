import pandas as pd


def load_zipcode_data():
    iou = pd.read_csv("data/iou_zipcodes_2023.csv")
    noniou = pd.read_csv("data/non_iou_zipcodes_2023.csv")
    zip_data = pd.concat([noniou, iou], ignore_index=True)
    zip_data['zip'] = zip_data['zip'].astype(str)
    return zip_data