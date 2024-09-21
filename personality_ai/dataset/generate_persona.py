'''Generates 5000 different persona. '''

import pandas as pd
import random

persona_options = {
    "Work": ["Unemployed", "Student", "Retired", "Part-time worker", "Full time employed", "Entrepreneur", "Contractor"],
    "Personality": ["Introvert", "Extrovert", "Ambivert", "Analytical", "Creative", "Driven", "Loyal", "Dependable", "Energetic"],
    "Gender": ["Male", "Female", "Unknown"],
    "Religion": ["Christianity", "Islam", "Hinduism", "Buddhism", "Sikhism", "Judaism", "no religion"],
    "Political": ["Liberal", "Conservative", "Moderate", "Libertarian", "Green", "Socialist", "Communist"],
    "Education": ["None", "Some High School", "High School Graduate", "Some College", "Associate Degree", "Bachelor's Degree", "Master's Degree", "Professional Degree", "Doctorate", "Trade School", "Certification", "Self-taught", "Online Courses", "Apprenticeship", "Continuing Education"],
    "Occupation": ["Healthcare", "Technology", "Education", "Business", "Retail", "Hospitality", "Manufacturing", "Construction", "Arts", "Science", "Government", "Agriculture", "Transportation", "Military", "Unemployed"],
    "Family": ["Single", "Married", "Divorced", "Widowed", "Engaged", "In a relationship"],
    "Fear": ["Failure", "Rejection", "Loneliness", "Death", "Illness", "Poverty", "Public Speaking", "Heights", "Darkness", "Intimacy", "Water", "Flying", "Spiders", "Snakes", "Clowns"],
    "Enjoys": ["Reading", "Writing", "Traveling", "Hiking", "Gaming", "Cooking", "Baking", "Photography", "Dancing", "Singing", "Playing musical instruments", "Watching movies", "Gardening", "Fishing", "Crafting"]
}








def generate_persona():
    return {category: random.choice(options) for category, options in persona_options.items()}

def generate_persona_descriptions(number_of_personas):
    personas = []
    for _ in range(number_of_personas):
        persona = generate_persona()

        persona['Age'] = random.randint(18, 70)
        pronoun = "They" if persona['Gender'] == "Unknown" else "He" if persona['Gender'] == "Male" else "She"
        
        description = f"{pronoun} is a {persona['Age']} yeard old {persona['Gender']} who is a {persona['Work']}. "\
                      f"Identifying as a {persona['Personality']} personality, {pronoun.lower()} were educated to {persona['Education']} level. "\
                      f"Politically, {pronoun.lower()} lean towards {persona['Political']} and follow {persona['Religion']}. "\
                      f"In their family, {pronoun.lower()} are {persona['Family']}, and {pronoun.lower()} fear {persona['Fear']}. "\
                      f"For fun, {pronoun.lower()} enjoy {persona['Enjoys']}."
        personas.append(description)
    return personas

# Generate 5 unique persona descriptions

if __name__ == "__main__":
    persona_descriptions = generate_persona_descriptions(5000)
    df = pd.DataFrame(persona_descriptions, columns=["Description"])
    df.to_csv('persona_descriptions.csv', index=False)
