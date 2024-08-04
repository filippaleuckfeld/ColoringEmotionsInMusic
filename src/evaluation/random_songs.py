import csv
import math
import random
from collections import defaultdict

def get_emotional_quadrant(valence, energy):
    # Adjust coordinates to the center (0.5, 0.5)
    adjusted_valence = valence - 0.5
    adjusted_energy = energy - 0.5

    # Calculate the angle in radians
    angle = math.atan2(adjusted_energy, adjusted_valence)
    
    angle_degrees = math.degrees(angle)
    
    if angle_degrees < 0:
        angle_degrees += 360

    if 0 <= angle_degrees < 45:
        return 1
    elif 45 <= angle_degrees < 90:
        return 2
    elif 90 <= angle_degrees < 135:
        return 3
    elif 135 <= angle_degrees < 180:
        return 4
    elif 180 <= angle_degrees < 225:
        return 5
    elif 225 <= angle_degrees < 270:
        return 6
    elif 270 <= angle_degrees < 315:
        return 7
    elif 315 <= angle_degrees < 360:
        return 8

def classify_isrcs_by_quadrant(csv_file):
    quadrants = defaultdict(list)  # Dictionary to hold lists of ISRCs for each quadrant

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            isrc = row['isrc']
            valence = float(row['predicted_valence_rf'])
            energy = float(row['predicted_energy_rf'])
            quadrant = get_emotional_quadrant(valence, energy)
            quadrants[quadrant].append(isrc)
    
    return quadrants

def get_random_isrc_per_quadrant(quadrants):
    random_isrcs = {}
    for quadrant, isrcs in quadrants.items():
        if isrcs:
            random_isrcs[quadrant] = random.choice(isrcs)
        else:
            random_isrcs[quadrant] = None
    return random_isrcs

def get_three_unique_numbers_excluding(exclude):
    numbers = list(range(1, 9))  # Create a list of numbers from 1 to 8
    numbers.remove(exclude)  # Remove the quadrant number
    selected_numbers = random.sample(numbers, 3)  # Randomly select 3 unique numbers
    return selected_numbers

csv_file = 'src/model/predicted_valence_energy_scores_rf.csv'
quadrants = classify_isrcs_by_quadrant(csv_file)
random_isrcs = get_random_isrc_per_quadrant(quadrants)

for quadrant in range(1, 9):
    isrc = random_isrcs[quadrant]
    if isrc:
        unique_numbers = get_three_unique_numbers_excluding(quadrant)
        print(f"Quadrant {quadrant}: ISRC: {isrc}, Three unique numbers (excluding {quadrant}): {unique_numbers}")
    else:
        print(f"Quadrant {quadrant}: No ISRC available")