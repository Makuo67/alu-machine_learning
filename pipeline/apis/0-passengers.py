#!/usr/bin/env python3
"""Returns a list of ships that can hold a given number of passengers"""
import requests


def availableShips(passengerCount):
    """
    Function to get request
    Args:
        passengerCount: number of passangers
    Returns: List of starships
    """
    starships = []
    url = 'https://swapi-api.alx-tools.com/api/starships/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'starships'})
        for ship in response.json()['results']:
            passenger = ship['passengers'].replace(',', '')
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                starships.append(ship['name'])
        url = response.json()['next']
    return starships
