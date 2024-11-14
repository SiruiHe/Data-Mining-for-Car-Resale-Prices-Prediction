import pandas as pd
import json
import time
import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def read_prompt(file_path='prompt.txt'):
    """Read prompt template from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_text_prompt(row, prompt_template):
    """Create prompt for each data row"""
    return prompt_template.format(
        title=row['title'] if pd.notna(row['title']) else '',
        description=row['description'] if pd.notna(row['description']) else '',
        features=row['features'] if pd.notna(row['features']) else '',
        accessories=row['accessories'] if pd.notna(row['accessories']) else ''
    )

async def get_gpt_response(client, prompt, max_retries=3, sleep_time=5):
    """Send async request to GPT API and handle response"""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes car listings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                print(f"JSON parsing error: {response_text}")
                return None
                
        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(sleep_time)
            else:
                return None

async def process_batch_async(df_batch, client, prompt_template, semaphore):
    """Process a batch of data asynchronously"""
    tasks = []
    for _, row in df_batch.iterrows():
        prompt = create_text_prompt(row, prompt_template)
        tasks.append(process_single_item(row, client, prompt, semaphore))
    
    results = await tqdm_asyncio.gather(*tasks)
    
    # Convert to DataFrame and ensure listing_id is the first column
    results_df = pd.DataFrame(results)
    cols = ['listing_id'] + [col for col in results_df.columns if col != 'listing_id']
    results_df = results_df[cols]
    
    return results_df

async def process_single_item(row, client, prompt, semaphore):
    """Process single data item with semaphore control"""
    async with semaphore:  # Use semaphore to control concurrency
        response = await get_gpt_response(client, prompt)
        
        if response is None:
            response = {
                "brand_popularity_score": 0.4,
                "model_value_score": 0.3,
                "condition_score": 0.5,
                "feature_rarity_score": 0.3,
                "performance_score": 0.5,
                "sentiment_score": 0.5
            }
        
        response['listing_id'] = row['listing_id']
        return response

async def extract_text_features_async(df, batch_size=20, output_file='data/with_text_features_temp.csv'):
    """Main async function: Process all data and return new features"""
    # Check for existing processed file
    processed_ids = set()
    if os.path.exists(output_file):
        print("Found existing results, loading processed data...")
        processed_df = pd.read_csv(output_file)
        processed_ids = set(processed_df['listing_id'])
        print(f"Processed {len(processed_ids)} records")
    
    # Filter unprocessed data
    df_to_process = df[~df['listing_id'].isin(processed_ids)]
    if len(df_to_process) == 0:
        print("All data has been processed!")
        return pd.read_csv(output_file)
    
    print(f"Need to process {len(df_to_process)} new records")
    
    # Initialize OpenAI client
    client = AsyncOpenAI()
    
    # Read prompt template
    prompt_template = read_prompt()
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(10)  # Process 10 requests simultaneously
    
    # Process data in batches
    all_results = []
    for i in range(0, len(df_to_process), batch_size):
        df_batch = df_to_process.iloc[i:i+batch_size]
        batch_results = await process_batch_async(df_batch, client, prompt_template, semaphore)
        all_results.append(batch_results)
        
        # Merge current batch results
        current_results_df = pd.concat(all_results, ignore_index=True)
        
        if os.path.exists(output_file):
            # If file exists, merge new and old results
            old_df = pd.read_csv(output_file)
            merged_df = pd.concat([
                old_df,
                current_results_df[~current_results_df['listing_id'].isin(old_df['listing_id'])]
            ])
            merged_df.to_csv(output_file, index=False)
        else:
            # If file doesn't exist, save directly
            current_results_df.to_csv(output_file, index=False)
        
        print(f"Saved batch results, progress: {i + len(df_batch)}/{len(df_to_process)}")
        await asyncio.sleep(0.5)
    
    # Read final results
    final_df = pd.read_csv(output_file)
    
    # Rename feature columns
    feature_cols = [col for col in final_df.columns if col != 'listing_id']
    final_df.columns = ['listing_id'] + ['text_' + col for col in feature_cols]
    
    return final_df

if __name__ == "__main__":
    import argparse
    
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='Mode to run: train or test (default: train)')
    args = parser.parse_args()
    
    # Set input and output file paths based on mode
    input_file = f'data/{args.mode}.csv'
    output_file = f'data/with_text_features_{args.mode}.csv'
    temp_file = f'data/with_text_features_temp_{args.mode}.csv'
    
    # Read original data
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Extract text features
    print("Starting text feature extraction...")
    text_features = asyncio.run(extract_text_features_async(df, output_file=temp_file))
    
    # Merge new features with original data
    df_with_features = df.merge(text_features, on='listing_id', how='left')
    
    # Save final results
    df_with_features.to_csv(output_file, index=False)
    print(f"Feature extraction completed! Final results saved to {output_file}")
    
    # Remove temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Temporary file {temp_file} has been removed")
    
    # Print statistics of new features
    feature_cols = [col for col in text_features.columns if col != 'listing_id']
    print("\nNew Feature Statistics:")
    print(text_features[feature_cols].describe())