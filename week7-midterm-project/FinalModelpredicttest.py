#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://127.0.0.1:5000/strokepredict/'

# sample person's details in JSON format - 

person_id = 'person-abc123'
sample_person = {"age": 75.0,"avg_glucose_level": 170.01,"bmi": 35.5,
"gender": "Male","hypertension": 0,"heart_disease": 1,"ever_married": "Yes",
"work_type": "private","residence_type": "Rural","smoking_status": "smokes"}

# sending this sample_person in a POST request to our web service - using post() function:
# To see the body of the response: takes the JSON response and converts it into a Python dictionary - 
response = requests.post(url, json=sample_person).json()

print(response)

# sending a message if response is stroke:
if response['stroke'] == True:
    print('person will have stroke %s' % person_id)
else:
    print('person will not have stroke %s' % person_id)




