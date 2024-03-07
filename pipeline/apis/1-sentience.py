#!/usr/bin/env python3
"""
Function that returns homes of planet sentient

"""
import requests


def sentientPlanets():
    """ Sentient Planets"""
    sentient = []
    url = "https://swapi-api.alx-tools.com/api/species/"

    while url:
        response = requests.get(url)
        result = response.json()

        for res in result['results']:
            if res['classification'] == "sentient" or \
                    res['designation'] == 'sentient':
                if res['homeworld'] is not None:
                    homeworld = requests.get(res['homeworld'], timeout=1000)
                    sentient.append(homeworld.json()['name'])

        url = result['next']

    return sentient
