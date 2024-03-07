#!/usr/bin/env python3
"""Returns a list of ships that can hold a given number of passengers"""


import requests


def availableShips(passengerCount):
    """API for available ships"""
    url = "https://swapi-api.alx-tools.com/api/"
    ships = []
    response = requests.get(f'{url}/starships')

    while response.status_code == 200:
        result = response.json()
        for ship in result['results']:
            passengers = ship['passengers'].replace(',', '')
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                pass

            try:
                response = requests.get(response['next'])
            except Exception:
                break
    return ships


availableShips(4)
