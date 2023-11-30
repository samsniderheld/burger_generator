import numpy as np
import random

from arg_parser import parse_prgen_args
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.basic_utils import read_ingredients_from_txt
from utils.pregen_utils import (count_normalized_lists, 
                                calculate_probabilities,
                                load_data_from_pkl,
                                save_recipes_pkl)



args = parse_prgen_args()



white_list = read_ingredients_from_txt(args.white_list_path)
all_ingredients = load_data_from_pkl(args.dataset_path)
embedder = SentenceTransformer(args.embedding_model_path)
ingredient_frequency_cutoff = args.ingredient_frequency_cutoff
num_recipes = args.num_recipes
min_rec_frequency = args.recipe_frequency_cuttoff
output_path = args.output_dir

white_list_embeds = embedder.encode(white_list)
all_ing_embeds = embedder.encode(all_ingredients)

# Calculate cosine similarity
cos_sim_matrix = cosine_similarity(all_ing_embeds, white_list_embeds)

# Find the index of the max cosine similarity for each row in array1
closest_indices = np.argmax(cos_sim_matrix, axis=1)

white_listed_ingredients = []

for i, ing in enumerate(all_ingredients):
  idx = closest_indices[i]
  white_listed_ingredient = white_list[idx]
  white_listed_ingredients.append(white_listed_ingredient )

#calculate the ingredient probablities given a min frequency cut off
probs,freqs = calculate_probabilities(white_listed_ingredients,
                                      ingredient_frequency_cutoff)

cleaned_ingredients = list(probs.keys())
cleaned_probs = list(probs.values())

sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

#generate the recipes
all_recipes = []
for i in tqdm(range(num_recipes)):
  #todo generate weighted random number based on dataset
  num_ing = random.randint(3,8)
  new_rec = random.choices(cleaned_ingredients, weights=cleaned_probs, k=num_ing)
  # makes sure no doubles are used
  while len(new_rec) != len(set(new_rec)):
    new_rec = random.choices(cleaned_ingredients, weights=cleaned_probs, k=num_ing)

  all_recipes.append(new_rec)

# Count the normalized lists
list_counts = count_normalized_lists(all_recipes)

pruned_recipes = {k:v for k, v in list_counts.items() if v > min_rec_frequency}

sorted_pruned_recipes = sorted(pruned_recipes.items(), key=lambda x: x[1], reverse=True)

# Print the counts
for ingredients, count in sorted_pruned_recipes[:100]:
    print(f"Ingredients: {ingredients}, Count: {count}")

save_recipes_pkl(sorted_pruned_recipes, output_path)