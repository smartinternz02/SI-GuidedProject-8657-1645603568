# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:39:33 2021

@author: HP
"""

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "94Sb4mTPK3QgBVQl-QFNydq33hoivC3SUYR9H4Z_8bhX"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line

payload_scoring = {"input_data": [{"fields": ["city","BHKS","sqft_per_inch","build_up_area","Type_of_property","deposit"], "values": [[5,1.0,470.0,1,2,11.225257]]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/21a2ec16-011e-4754-8228-cb02cfaaeb6d/predictions?version=2021-12-15', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())