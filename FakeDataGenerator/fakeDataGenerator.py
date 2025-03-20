import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

records_list = []
person_counter = 1000000  # Starting PersonLopNr

# Pre-calculate profession ranges for 50 unique professions (3310 to 3359)
professions = list(range(3310, 3310+50))  # exactly 50 unique professions
profession_ranges = {}
for i, prof in enumerate(professions):
    p_lower = 500 + i * 300         # e.g., for first: 500, then 800, 1100, ...
    p_upper = p_lower + 5000          # fixed width of 5000 for the income range
    profession_ranges[prof] = (p_lower, p_upper)

# Pre-calculate university ranges for 30 unique universities (uni_id 1 to 30)
uni_ids = list(range(1, 31))
uni_ranges = {}
for j, uid in enumerate(uni_ids):
    u_lower = 1000 + j * 400        # e.g., for first: 1000, then 1400, 1800, ...
    u_upper = u_lower + 6000         # fixed width of 6000 for the income range
    uni_ranges[uid] = (u_lower, u_upper)

# Education values from sample file (e.g., 337 and 441)
education_values = [337, 441, 640, 536, 729]

# List of 25 largest Swedish cities with approximate populations (2021 data)
cities_with_pop = {
    "Stockholm": 975551,
    "Gothenburg": 579281,
    "Malmo": 347949,
    "Uppsala": 233839,
    "Vasteras": 154049,
    "Orebro": 155989,
    "Linkoping": 163051,
    "Helsingborg": 149280,
    "Jönköping": 142427,
    "Norrkoping": 143171,
    "Lund": 124935,
    "Umeå": 130224,
    "Borås": 113179,
    "Eskilstuna": 106859,
    "Huddinge": 113951,
    "Sundbyberg": 53564,
    "Sollentuna": 74041,
    "Tumba": 48057,
    "Trollhattan": 59249,
    "Växjö": 94859,
    "Karla": 45000,  # Approximate
    "Upplands Vasby": 47820,
    "Alingsås": 41602,
    "Malmö": 347949  # Alternative spelling included
}

# Create lists for random.choices
cities = list(cities_with_pop.keys())
weights = list(cities_with_pop.values())

# Ethnicity: using values from sample (only "SE")
ethnicities = ["SE", "NW", "W", "A", "AS"]

# Gender options: only "m" or "f"
genders = ["m", "f"]

total_records_needed = 100000

# Add these constants near the top after the imports
INCOME_LOG_MEAN = 8.5  # This gives median income around 5000 (exp(8.5) ≈ 5000)
INCOME_LOG_STD = 0.6   # This gives reasonable spread in the distribution

def create_income_modifiers():
    """Create stable random modifiers for higher-order interactions"""
    # Base modifiers
    city_size_effect = 0.3  # Positive effect of city size
    gender_effect = {'m': 0.15, 'f': -0.15}  # Men earn more
    parental_effect = 0.25  # Positive effect of parental income
    education_effect = 0.2  # Positive effect of education level
    
    # Create stable random interaction effects
    np.random.seed(42)  # Ensure reproducibility
    
    # Generate higher-order interaction matrices
    # City-Parent interaction (20x10 matrix for city size quintile x parent income decile)
    city_parent_interaction = np.random.normal(0, 0.1, (20, 10))
    
    # Education-Gender-Parent interaction (5x2x10 tensor)
    edu_gender_parent = np.random.normal(0, 0.05, (5, 2, 10))
    
    # City-Education interaction (20x5 matrix)
    city_education_interaction = np.random.normal(0, 0.07, (20, 5))
    
    return {
        'city_size': city_size_effect,
        'gender': gender_effect,
        'parental': parental_effect,
        'education': education_effect,
        'city_parent': city_parent_interaction,
        'edu_gender_parent': edu_gender_parent,
        'city_education': city_education_interaction
    }

def calculate_income_multiplier(city, gender, parental_income, education, modifiers, cities_with_pop):
    """Calculate income multiplier based on characteristics and their interactions"""
    # Base multiplier
    multiplier = 1.0
    
    # City size effect (normalized by largest city)
    city_size_rank = sorted(cities_with_pop.values(), reverse=True).index(cities_with_pop[city])
    city_quintile = city_size_rank // 5
    max_pop = max(cities_with_pop.values())
    multiplier += modifiers['city_size'] * (cities_with_pop[city] / max_pop)
    
    # Gender effect
    multiplier += modifiers['gender'][gender]
    
    # Parental income effect (using deciles)
    parent_decile = min(parental_income // 10, 9)  # Ensure it doesn't exceed 9 (0-9 for 10 deciles)
    multiplier += modifiers['parental'] * (parental_income / 100)
    
    # Education effect (normalized)
    edu_index = sorted(education_values).index(education)
    multiplier += modifiers['education'] * (edu_index / len(education_values))
    
    # Higher-order interactions
    # City-Parent interaction
    multiplier += modifiers['city_parent'][city_quintile, parent_decile]
    
    # Education-Gender-Parent interaction
    gender_idx = 0 if gender == 'm' else 1
    multiplier += modifiers['edu_gender_parent'][edu_index, gender_idx, parent_decile]
    
    # City-Education interaction
    multiplier += modifiers['city_education'][city_quintile, edu_index]
    
    return max(0.5, min(2.0, multiplier))  # Constrain multiplier between 0.5 and 2.0

# Create income modifiers once
income_modifiers = create_income_modifiers()

while len(records_list) < total_records_needed:
    # Create a new individual
    person_id = person_counter
    person_counter += 1
    
    gender = random.choice(genders)
    education = random.choice(education_values)
    city = random.choices(cities, weights=weights, k=1)[0]
    ethnicity = random.choice(ethnicities)
    parental_income = random.randint(0, 100)  # parental_income_percentile7
    
    # Randomly assign a profession and uni_id
    profession = random.choice(professions)
    uni_id = random.choice(uni_ids)
    
    # Get income ranges from profession and university
    p_lower, p_upper = profession_ranges[profession]
    u_lower, u_upper = uni_ranges[uni_id]
    allowed_lower = max(p_lower, u_lower)
    allowed_upper = min(p_upper, u_upper)
    
    # Get base log-income from lognormal distribution
    base_log_income = np.random.normal(INCOME_LOG_MEAN, INCOME_LOG_STD)
    base_income = np.exp(base_log_income)
    
    # Calculate income multiplier based on characteristics
    income_mult = calculate_income_multiplier(
        city, gender, parental_income, education, 
        income_modifiers, cities_with_pop
    )
    
    # Apply multiplier to base income
    base_income *= income_mult
    
    # Choose starting age and starting year with constraints:
    start_age = random.randint(20, 65)
    start_year = random.randint(1980, 2025)
    max_records_age = 65 - start_age + 1
    max_records_year = 2025 - start_year + 1
    max_possible_records = min(max_records_age, max_records_year, 5)  # cap sequence length at 5
    
    seq_length = random.randint(1, max_possible_records)
    
    # Generate income progression for this individual
    if seq_length == 1:
        incomes = [base_income]
    else:
        # Create log-space progression
        growth_rate = random.uniform(0.01, 0.05)  # 1-5% annual growth
        years = np.arange(seq_length)
        income_progression = base_income * np.exp(growth_rate * years)
        incomes = income_progression.tolist()
    
    # Create sequential records for the individual
    for i in range(seq_length):
        record = {
            "PersonLopNr": person_id,
            "income": round(incomes[i], 2),
            "profession": profession,
            "gender": gender,
            "education": education,
            "city": city,
            "uni_id": uni_id,
            "ethnicity": ethnicity,
            "parental_income_percentile7": parental_income,
            "age": start_age + i,
            "year": start_year + i,
            "cohort": (start_year - start_age)  # remains constant for the individual
        }
        records_list.append(record)
        if len(records_list) >= total_records_needed:
            break

# Add data validation after generating records
df = pd.DataFrame(records_list).head(total_records_needed)

# Add validation checks
print(f"Data validation:")
print(f"Number of unique individuals: {df['PersonLopNr'].nunique()}")
print(f"Age range: {df['age'].min()} to {df['age'].max()}")
print(f"Income range: {df['income'].min():.2f} to {df['income'].max():.2f}")
print(f"Gender distribution:\n{df['gender'].value_counts(normalize=True)}")

# Save the DataFrame to a CSV file
csv_file = "large_data.csv"
df.to_csv(csv_file, index=False)
print(f"\nGenerated {len(df)} records and saved to {csv_file}.")