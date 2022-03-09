import requests
from pandas.io.json import json_normalize
import wapi
import pytz
import os

def get_data_mdap(id,hyp,timeres):

    ## PAT to authenticate against azure active directory when using MDAP-calls
    access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI4YTllOTNlYi05ZTRmLTQ0OWUtOTgyMi1hNWZlM2Y5YTJkZmIiLCJh' \
                   'cHBpZCI6IlM0OTM5OUB1bmlwZXIuZW5lcmd5IiwibmJmIjoxNjA1Mjc3ODA1LCJleHAiOjE2MjEyNjE4MDQsImlhdCI6MTYwNTI3N' \
                   'zgwNSwiaXNzIjoibWRhcF9kYXRhX3NlcnZpY2VfYXBwbGljYXRpb24ifQ.acdkl90ec34SMBjlGLwax2GYwaUZ48U5V9A3eLRZGJs'

    # case dependant variables
    sCase = 'PROD'
    if sCase == 'UAT':
        ENDPOINT = 'https://uniper-mdap-data-service-uat.azurewebsites.net/api/time-series-data'
    elif sCase == 'PROD':
        ENDPOINT = 'https://uniper-mdap-data-service-prd.azurewebsites.net/api/time-series-data'


    ## starting to build the header set for MDAP-API
    headers_set = {}
    headers_set['Authorization'] = 'Bearer ' + access_token
    headers_set['Content-Type'] = 'application/json'
    headers_set['x-component-id'] = 'test'
    headers_set['x-correlation-id'] = 'test'
    headers_set['charset'] = 'ascii'
    # headers_set['Host']="uniper-mdap-data-service-prd.azurewebsites.net"

    params_set = {}
    # defining the data container
    params_set['timeseriesId'] = id
    params_set['hypothesis'] = hyp
    params_set['timeresolution'] = timeres
    params_set['fromTimepoint'] = '2018-01-01 00:00:00'

    # technical parameters (limit the number of returned results)
    # params_set['limit'] = 10000
    # params_set['offset']=0

    # select a sub-set of columns
    # params_set['fields']='Contract_Size'

    # apply a row filter (this is technically equivalent to specifying a where-statement in SQL)
    #params_set['filter'] = "FullProductName='ENOAFUTBLMAUG-19'"




    responseData = requests.request("GET", ENDPOINT, headers=headers_set, params=params_set)

    ## checking the response
    if responseData.status_code == 200:
        json_output = responseData.json()
        print('successfull request at endpoint {0}'.format(responseData.url))
        print('number of items retrieved: ' + str(len(json_output)))
        df = json_normalize(json_output)
    else:
        print('error in request')
        print(responseData.text)
        print(responseData.status_code)
        print(responseData.url)
        raise AttributeError("MDAP API replied an invalid request")

    return df

#


def get_curves_wattsight(inputs,credentials):

    session = wapi.Session(client_id= credentials[0], client_secret= credentials[1])

    curves = session.search(commodity= inputs['commodity'],
                            category=inputs['category'],
                            unit=inputs['units'],
                            source=inputs['source'],
                            area=inputs['area'],
                            data_type=inputs['data_type']
                            )
    return curves

def get_data_wattsight(curves, issue_dates, date_from = None, date_to = None):

    data = {}

    for c in curves:
        data[c.name] = {}



        if 'INSTANCES' in c.curve_type:

            print('---------------------processing instance : ', c.name)
            for d in issue_dates:

                ts = c.get_instance(d)

                if ts is not None:
                    data[c.name][d.strftime('%Y-%m-%d %H:%M:%S')] = ts.to_pandas()
                    print('Added : ', d)

        elif c.curve_type == 'TIME_SERIES':

            print('------------------ processing time series : ',c.name)

            data[c.name] = c.get_data(data_from=date_from,
                                      data_to=date_to).to_pandas()

    return data

