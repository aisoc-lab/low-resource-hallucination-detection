import litellm as openai  # Use litellm for the OpenAI API
import pandas as pd
import csv

# Set up your OpenAI API key
openai.api_key = 'your_api_key_here'

# Cache to store translations to avoid redundant API calls
translation_cache = {}
api_call_count = 0  # To track the number of API calls made

# Function to translate text using litellm's ChatCompletion
def translate_text(text):
    global api_call_count  # Access global variable to track API calls
    normalized_text = text.strip().lower().replace(' language', '')  # Normalize by removing "language" for official language dataset

    # Check if the text is already in the cache
    if text in translation_cache:
        return translation_cache[text]
    
    try:
        response = openai.completion(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Translate the following text to Urdu: {text}. The output should not be a sentence. It should only contain the translated text.",
            }],
            max_tokens=40,
            temperature=0.1
        )
        translated_text = response['choices'][0]['message']['content'].strip().replace('\n', '-')
        
        # Store translation in cache and increment API call count
        translation_cache[normalized_text] = translated_text
        api_call_count += 1
        
        return translated_text
    except Exception as e:
        raise Exception(f"Translation error: {e}")

# Function to translate the CSV file contents
def translate_csv_file(input_file, output_file, max_output_entries=2500):
    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create a list to hold translations
    translations = []
    c = 0
    for index, row in df.iterrows():
        # Stop if we reach the max output entries limit
        if len(translations) >= max_output_entries:
            print(f"Reached the maximum output limit of {max_output_entries} entries.")
            break

        c += 1
        try:
            if c % 10 == 0:
                print(f"Processed till row {c}...")

            # Get the English text from the two columns
            english_place = row[0].replace('"', '').strip()  # Remove quotes and trim spaces
            row[1] = row[1].replace('"', '')
            english_targets = row[1].split('<OR>')  # Split targets by "<OR>"

            # Translate each target separately and join them back with "<OR>"
            source_translation = translate_text(english_place)  # Translate the source place name
            translated_targets = []

            for target in english_targets:
                target = target.strip()  # Normalize each target
                split_translation = translate_text(target)  # Translate each option separately

                # Remove '.' if it is the last character
                if split_translation.endswith('.'):
                    split_translation = split_translation[:-1]

                translated_targets.append(split_translation)

            # Join translated targets with "<OR>"
            joined_translations = ' <OR> '.join(translated_targets)

            # Append the translated pair of source and targets
            translations.append([source_translation, joined_translations])

        except Exception as e:
            print(f"Error processing row {c}: {e}")
            continue

    # Create a new DataFrame for the German translations
    translated_df = pd.DataFrame(translations, columns=['source', 'target'])

    try:
        # Save the translated content into a CSV file without any extra quotes or backslashes
        translated_df.to_csv(output_file, index=False, encoding='utf-8', sep=',')
        print(f"Translation complete. Check the output file: {output_file}")
        print(f"Total number of API calls made: {api_call_count}")
    except Exception as e:
        print(f"Error saving translated CSV: {e}")

# Specify your input and output file paths
input_file = 'capitals.csv'  # Replace with your actual input file
output_file = 'capitals_openai_urdu.csv'  # The output file with translations

# Run the translation process with a 2500-entry output limit
translate_csv_file(input_file, output_file, max_output_entries=2500)
