import pickle
from collections import Counter

def load_data_from_pkl(path):
    #loads the ingredient data
    with open(path,'rb') as f:
        data = pickle.load(f)

    return data

def save_recipes_pkl(recipes,path):
    with open(path,'wb') as f:
        pickle.dump(recipes,f)

def count_normalized_lists(lists):
    #Function to normalize and count lists

    # Normalize each list by sorting and converting to tuple
    normalized_lists = [tuple(sorted(ingredient_list)) for ingredient_list in lists]

    # Count occurrences of each normalized list
    return Counter(normalized_lists)

def calculate_probabilities(ingredients, min_freq):
    # Count the frequency of each ingredient
    frequency = {}
    for ingredient in ingredients:
        if ingredient in frequency:
            frequency[ingredient] += 1
        else:
            frequency[ingredient] = 1

    frequency = {k:v for k, v in frequency.items() if v > min_freq}

    # Total number of ingredients
    total_ingredients = sum(frequency.values())

    # Calculate probability for each ingredient
    probabilities = {ingredient: freq / total_ingredients for ingredient, freq in frequency.items()}
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))


    return sorted_probabilities, frequency