import random
import pandas as pd


names_genders = [
    ("James", "Male"), ("Mary", "Female"), ("John", "Male"), ("Patricia", "Female"),
    ("Robert", "Male"), ("Jennifer", "Female"), ("Michael", "Male"), ("Linda", "Female"),
    ("William", "Male"), ("Elizabeth", "Female"), ("David", "Male"), ("Barbara", "Female"),
    ("Richard", "Male"), ("Susan", "Female"), ("Joseph", "Male"), ("Jessica", "Female"),
    ("Thomas", "Male"), ("Sarah", "Female"), ("Charles", "Male"), ("Karen", "Female"),
    ("Christopher", "Male"), ("Nancy", "Female"), ("Daniel", "Male"), ("Lisa", "Female"),
    ("Matthew", "Male"), ("Betty", "Female"), ("Anthony", "Male"), ("Margaret", "Female"),
    ("Mark", "Male"), ("Sandra", "Female")
]

countries = [
    "the United States", "Canada", "the United Kingdom", "Australia", "Germany", 
    "France", "Japan", "China", "India", "Brazil"
]



occupations = [
    "Software Developer", "Teacher", "Doctor", "Nurse", "Engineer", "Artist", "Musician", "Chef", 
    "Photographer", "Writer", "Journalist", "Lawyer", "Accountant", "Architect", "Scientist", 
    "Pharmacist", "Dentist", "Veterinarian", "Pilot", "Flight Attendant", "Police Officer", 
    "Firefighter", "Mechanic", "Electrician", "Plumber", "Carpenter", "Farmer", "Librarian", 
    "Psychologist", "Social Worker"
]

big_five_personalities = {
    "Openness": ["open to new experiences", "highly imaginative", "not creative"],
    "Conscientiousness": ["very driven",  "organized", "not dependable"],
    "Extraversion": ["extremely outgoing", "energetic", "quite reserved"],
    "Agreeableness": ["loyal", "very compassionate", "extremely critical"],
    "Neuroticism": ["very anxious", "calm", "highly sensitive"]
}

personalities = [f"{op}, {con}, {ext}, {agr}, {neu}" for op in big_five_personalities["Openness"]
                 for con in big_five_personalities["Conscientiousness"]
                 for ext in big_five_personalities["Extraversion"]
                 for agr in big_five_personalities["Agreeableness"]
                 for neu in big_five_personalities["Neuroticism"]]

scenarios = [
    "encounters a friend:",
    "goes to work:",
    "attends a meeting:",
    "visits a new city:",
    "goes shopping:",
    "attends a concert:",
    "goes to a party:",
    "takes a walk in the park:",
    "goes to the gym:",
    "visits a museum:",
    "goes to a restaurant:",
    "travels by plane:",
    "attends a workshop:",
    "goes to a festival:",
    "goes to a nightclub:",
    "visits a farm:",
    "goes to a zoo:",
    "goes to a theme park:",
    "attends a religious service:",
    "thinks about her future:",
    "shares her opinion on politics:",
    "discusses her favorite books:",
    "talks about her favorite movies:",
    "shares her thoughts on climate change:",
    "expresses her taste in music:",
    "talks about her favorite hobbies:"
]

def generate_character_description(scenario):
    name, gender = random.choice(names_genders)
    country = random.choice(countries)
    age = random.randint(18, 70)
    occupation = random.choice(occupations)
    personality = random.choice(personalities)

    pronoun = "They" if gender == "Unknown" else "He" if gender == "Male" else "She"
    description = f"{name} is a {age}-year-old {gender.lower()} from {country}. {pronoun} works as a {occupation} and is {personality}. "\
                  f"{name} now {scenario.lower()}"

    return description


if __name__ == "__main__":
    number_of_characters = 1000
    selected_scenario = random.choice(scenarios)
    character_descriptions = [generate_character_description(selected_scenario) for _ in range(number_of_characters)]
    df = pd.DataFrame(character_descriptions, columns=["Description"])
    df.to_csv('character_descriptions.csv', index=False)
