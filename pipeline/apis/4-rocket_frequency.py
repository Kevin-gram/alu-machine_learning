#!/usr/bin/env python3
"""Script that displays the number of launches per rocket"""


import requests
from collections import defaultdict


def get_launches_per_rocket():
    """Script that displays the number of launches per rocket"""

    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets'

    try:
        launches_response = requests.get(launches_url)
        launches_response.raise_for_status()
        launches = launches_response.json()

        launch_count = defaultdict(int)
        for launch in launches:
            rocket_id = launch['rocket']
            launch_count[rocket_id] += 1

        rockets_response = requests.get(rockets_url)
        rockets_response.raise_for_status()
        rockets = rockets_response.json()

        rocket_names = {rocket['id']: rocket['name'] for rocket in rockets}

        rocket_launches = [
            (rocket_names[rocket_id], count)
            for rocket_id, count in launch_count.items()
        ]

        rocket_launches.sort(key=lambda x: (-x[1], x[0]))

        for rocket, count in rocket_launches:
            print("{}: {}".format(rocket, count))

    except requests.RequestException as e:
        print(
            'An error occurred while making an API request: {}'.format(e))
    except Exception as err:
        print('A general error occurred: {}'.format(err))


if __name__ == '__main__':
    get_launches_per_rocket()
