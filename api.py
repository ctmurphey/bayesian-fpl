#!/usr/bin/python3
# C.T. Murphey 2025-05-23
import json
import requests
 
# descriptions taken from https://medium.com/analytics-vidhya/getting-started-with-fantasy-premier-league-data-56d3b9be8c32
root = "https://fantasy.premierleague.com/api/"
main = root+"bootstrap-static/" # Main URL for all premier league players, teams, global gameweek summaries
fixtures = root+"fixtures/" # A list of all 380 matches that will happen over the season
players = root+"element-summary/" # Remaining fixtures left for PL player as well as previous fixtures and seasons
managers = root+"entry/" # entry/{TID}/ 	Basic info on FPL Manager
leagues = root+"leagues-classic/" # leagues-classic/{TID}/standings/ 	Information about league with id LID for leagues with more than 50 teams
stats = root+"event/" # event/{GW}/live/   Stats of all PL players that played in GW

mo_id = 328 # Mo Salah's playerID for testing purposes
# print(requests.get(players+str(mo_id)+"/").json()['history'][15])

# print(requests.get(players+"2/").json().keys())
# print(requests.get(main).json()['elements'][462]['first_name'])
print(requests.get(stats+str(17)+"/live/").json()['elements'][mo_id])#[str(mo_id)])
def get_name(id):
    return
