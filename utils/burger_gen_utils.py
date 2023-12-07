"""
Burger Generation Utility Functions

This module offers utility functions to help generate burger images.
- Enforcing a 2:1 standard ingredient to random ingredient generation. This helps visual quality.
- Constructing the prompt and negative prompt from a list of ingredients.
- Chunking prompts so that they are encoded more like Automatic1111

"""
import random
import re
import torch

def enforce_standard_ingredient_ratio(all_ingredients, standard_ingredients, ingredients_num):
    """enforces a 2:1 BK ingredients to random ingredients ratio"""
    total_count = ingredients_num

    if total_count  == 1:
        standard_count = 1
        random_count = 0
    elif total_count ==2:
        standard_count = 1
        random_count = 1
    else:
        standard_count = 2 * total_count // 3
        random_count = total_count - standard_count
    
    selected_standard = random.sample(standard_ingredients, standard_count)
    selected_random = random.sample(all_ingredients, random_count)
    
    ingredients = selected_standard + selected_random

    return ingredients

def contstruct_prompt_from_ingredient_list(ingredients):
    """create our prompt for stable diffusion"""
    ingredient_string = "".join([f"({ingredient})++, " for ingredient in ingredients])
    prompt = f'(A whopper with a beef patty and {len(ingredients)} extra ingredients). {ingredient_string[:-1]}. On a white background.'
    return prompt

# def contstruct_prompt_from_ingredient_list(ingredients):
#     """create our prompt for stable diffusion"""
#     ingredient_string = "".join([f"'{ingredient}', " for ingredient in ingredients])[:-2]
#     val = 8/len(ingredients)
#     ingredient_weights = "".join([f"{val}, " for ingredient in ingredients])[:-2]
#     prompt = f"""('A whopper with a beef patty and {len(ingredients)} extra ingredients.', {ingredient_string[:-1]}).blend(1,{ingredient_weights})"""
#     return prompt

def construct_negative_prompt_for_standard_ingredients(ingredients,standard_ingredients):
    """create our negative prompt for stable diffusion if
    we are using standard ingredients, make sure that we are 
    negatively prompting correctly"""
    new_basic_ingredients = []

    for ing in standard_ingredients:
            if(ing not in ingredients):
                new_basic_ingredients.append(ing)

    negative_prompt = "poor quality, unappetizing, bad edges,  " + "".join([f"({ing})++, " for ing in new_basic_ingredients])

    return negative_prompt

#via https://github.com/damian0815/compel/issues/59
#and https://gist.github.com/tg-bomze/581a7e4014594609969d5ce8f0759b46
def parse_prompt_attention(text):
    re_attention = re.compile(r"""
      \\\(|
      \\\)|
      \\\[|
      \\]|
      \\\\|
      \\|
      \(|
      \[|
      :([+-]?[.\d]+)\)|
      \)|
      ]|
      [^\\()\[\]:]+|
      :
      """, re.X)

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re.compile(r"\s*\bBREAK\b\s*", re.S), text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

def prompt_attention_to_invoke_prompt(attention):
    tokens = []
    for text, weight in attention:
        # Round weight to 2 decimal places
        weight = round(weight, 2)
        if weight == 1.0:
            tokens.append(text)
        elif weight < 1.0:
            if weight < 0.8:
                tokens.append(f"({text}){weight}")
            else:
                tokens.append(f"({text})-" + "-" * int((1.0 - weight) * 10))
        else:
            if weight < 1.3:
                tokens.append(f"({text})" + "+" * int((weight - 1.0) * 10))
            else:
                tokens.append(f"({text}){weight}")
    return "".join(tokens)

def concat_tensor(t):
    t_list = torch.split(t, 1, dim=0)
    t = torch.cat(t_list, dim=1)
    return t

def merge_embeds(prompt_chanks, compel):
    num_chanks = len(prompt_chanks)
    if num_chanks != 0:
        power_prompt = 1/(num_chanks*(num_chanks+1)//2)
        prompt_embs,pooled = compel(prompt_chanks)
        t_list = list(torch.split(prompt_embs, 1, dim=0))
        for i in range(num_chanks):
            t_list[-(i+1)] = t_list[-(i+1)] * ((i+1)*power_prompt)
        prompt_emb = torch.stack(t_list, dim=0).sum(dim=0)
    else:
        prompt_emb = compel('')
    return prompt_emb,pooled

def detokenize(chunk, actual_prompt):
    chunk[-1] = chunk[-1].replace('</w>', '')
    chanked_prompt = ''.join(chunk).strip()
    while '</w>' in chanked_prompt:
        if actual_prompt[chanked_prompt.find('</w>')] == ' ':
            chanked_prompt = chanked_prompt.replace('</w>', ' ', 1)
        else:
            chanked_prompt = chanked_prompt.replace('</w>', '', 1)
    actual_prompt = actual_prompt.replace(chanked_prompt,'')
    return chanked_prompt.strip(), actual_prompt.strip()

def tokenize_line(line, tokenizer): # split into chunks
    actual_prompt = line.lower().strip()
    actual_tokens = tokenizer.tokenize(actual_prompt)
    max_tokens = tokenizer.model_max_length - 2
    comma_token = tokenizer.tokenize(',')[0]

    chunks = []
    chunk = []
    for item in actual_tokens:
        chunk.append(item)
        if len(chunk) == max_tokens:
            if chunk[-1] != comma_token:
                for i in range(max_tokens-1, -1, -1):
                    if chunk[i] == comma_token:
                        actual_chunk, actual_prompt = detokenize(chunk[:i+1], actual_prompt)
                        chunks.append(actual_chunk)
                        chunk = chunk[i+1:]
                        break
                else:
                    actual_chunk, actual_prompt = detokenize(chunk, actual_prompt)
                    chunks.append(actual_chunk)
                    chunk = []
            else:
                actual_chunk, actual_prompt = detokenize(chunk, actual_prompt)
                chunks.append(actual_chunk)
                chunk = []
    if chunk:
        actual_chunk, _ = detokenize(chunk, actual_prompt)
        chunks.append(actual_chunk)

    return chunks

def chunk_embeds(prompt, pipeline, compel):
    attention = parse_prompt_attention(prompt)
    global_attention_chanks = []

    for att in attention:
        for chank in att[0].split(','):
            temp_prompt_chanks = tokenize_line(chank, pipeline.tokenizer)
            for small_chank in temp_prompt_chanks:
                temp_dict = {
                    "weight": round(att[1], 2),
                    "lenght": len(pipeline.tokenizer.tokenize(f'{small_chank},')),
                    "prompt": f'{small_chank},'
                }
                global_attention_chanks.append(temp_dict)

    max_tokens = pipeline.tokenizer.model_max_length - 2
    global_prompt_chanks = []
    current_list = []
    current_length = 0
    for item in global_attention_chanks:
        if current_length + item['lenght'] > max_tokens:
            global_prompt_chanks.append(current_list)
            current_list = [[item['prompt'], item['weight']]]
            current_length = item['lenght']
        else:
            if not current_list:
                current_list.append([item['prompt'], item['weight']])
            else:
                if item['weight'] != current_list[-1][1]:
                    current_list.append([item['prompt'], item['weight']])
                else:
                    current_list[-1][0] += f" {item['prompt']}"
            current_length += item['lenght']
    if current_list:
        global_prompt_chanks.append(current_list)

    return merge_embeds([prompt_attention_to_invoke_prompt(i) for i in global_prompt_chanks], compel)