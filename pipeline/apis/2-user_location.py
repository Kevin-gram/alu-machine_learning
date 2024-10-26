#!/usr/bin/env python3
""" Prints the location of a GitHub user specified
as an arg to the script
"""
import sys
import requests
import json

if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 200:
        data = json.loads(response.text)
        print(data['location'])
    elif response.status_code == 403:
        print('Reset in ')
    elif response.status_code == 404:
        print('Not found')
    else:
        print('Error:', response.status_code)
