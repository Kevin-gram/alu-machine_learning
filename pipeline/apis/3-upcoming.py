#!/usr/bin/env python3
"""Script that displays the upcoming launch"""


import requests
import datetime


def get_upcoming_launch():
    """Script that displays the upcoming launch"""

    url = 'https://api.spacexdata.com/v4/launches/upcoming'

    try:
        response = requests.get(url)
        response.raise_for_status()
        launches = response.json()

        upcoming_launch = sorted(launches, key=lambda x: x['date_unix'])[0]

        rocket_id = upcoming_launch['rocket']
        rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
            rocket_id)
        rocket_response = requests.get(rocket_url)
        rocket_response.raise_for_status()
        rocket = rocket_response.json()
        rocket_name = rocket['name']

        launchpad_id = upcoming_launch['launchpad']
        launchpad_url = 'https://api.spacexdata.com/v4/launchpads/{}'.format(
            launchpad_id)
        launchpad_response = requests.get(launchpad_url)
        launchpad_response.raise_for_status()
        launchpad = launchpad_response.json()
        launchpad_name = launchpad['name']
        launchpad_locality = launchpad['locality']

        date_local = upcoming_launch['date_local']

        print(
            "{} ({}) {} - {} ({})".format(
                upcoming_launch['name'], date_local, rocket_name,
                launchpad_name, launchpad_locality))

    except requests.RequestException as e:
        print('An error occurred while making an API request: {}'.format(e))
    except Exception as err:
        print('A general error occurred: {}'.format(err))


if __name__ == '__main__':
    get_upcoming_launch()
