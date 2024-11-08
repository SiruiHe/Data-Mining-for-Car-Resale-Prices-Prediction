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
    """读取prompt模板"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_text_prompt(row, prompt_template):
    """为每行数据创建prompt"""
    return prompt_template.format(
        title=row['title'] if pd.notna(row['title']) else '',
        description=row['description'] if pd.notna(row['description']) else '',
        features=row['features'] if pd.notna(row['features']) else '',
        accessories=row['accessories'] if pd.notna(row['accessories']) else ''
    )

async def get_gpt_response(client, prompt, max_retries=3, sleep_time=5):
    """异步发送请求到GPT API并处理响应"""
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
                print(f"JSON解析错误: {response_text}")
                return None
                
        except Exception as e:
            print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(sleep_time)
            else:
                return None

async def process_batch_async(df_batch, client, prompt_template, semaphore):
    """异步处理一批数据"""
    tasks = []
    for _, row in df_batch.iterrows():
        prompt = create_text_prompt(row, prompt_template)
        tasks.append(process_single_item(row, client, prompt, semaphore))
    
    results = await tqdm_asyncio.gather(*tasks)
    
    # 转换为DataFrame并确保listing_id在第一列
    results_df = pd.DataFrame(results)
    cols = ['listing_id'] + [col for col in results_df.columns if col != 'listing_id']
    results_df = results_df[cols]
    
    return results_df

async def process_single_item(row, client, prompt, semaphore):
    """处理单个数据项"""
    async with semaphore:  # 使用信号量控制并发
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

async def extract_text_features_async(df, batch_size=20, output_file='data/cars_with_text_features.csv'):
    """异步主函数：处理所有数据并返回新特征"""
    # 检查是否存在已处理的文件
    processed_ids = set()
    if os.path.exists(output_file):
        print("发现已存在的处理结果，加载已处理的数据...")
        processed_df = pd.read_csv(output_file)
        processed_ids = set(processed_df['listing_id'])
        print(f"已处理 {len(processed_ids)} 条记录")
    
    # 过滤出未处理的数据
    df_to_process = df[~df['listing_id'].isin(processed_ids)]
    if len(df_to_process) == 0:
        print("所有数据都已处理完成！")
        return pd.read_csv(output_file)
    
    print(f"需要处理 {len(df_to_process)} 条新记录")
    
    # 初始化OpenAI客户端
    client = AsyncOpenAI()
    
    # 读取prompt模板
    prompt_template = read_prompt()
    
    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(10)  # 同时处理10个请求
    
    # 分批处理数据
    all_results = []
    for i in range(0, len(df_to_process), batch_size):
        df_batch = df_to_process.iloc[i:i+batch_size]
        batch_results = await process_batch_async(df_batch, client, prompt_template, semaphore)
        all_results.append(batch_results)
        
        # 合并当前批次结果
        current_results_df = pd.concat(all_results, ignore_index=True)
        
        if os.path.exists(output_file):
            # 如果文件存在，合并新旧结果
            old_df = pd.read_csv(output_file)
            merged_df = pd.concat([
                old_df,
                current_results_df[~current_results_df['listing_id'].isin(old_df['listing_id'])]
            ])
            merged_df.to_csv(output_file, index=False)
        else:
            # 如果文件不存在，直接保存
            current_results_df.to_csv(output_file, index=False)
        
        print(f"已保存批次结果，完成度: {i + len(df_batch)}/{len(df_to_process)}")
        await asyncio.sleep(0.5)
    
    # 读取最终结果
    final_df = pd.read_csv(output_file)
    
    # 重命名特征列
    feature_cols = [col for col in final_df.columns if col != 'listing_id']
    final_df.columns = ['listing_id'] + ['text_' + col for col in feature_cols]
    
    return final_df

if __name__ == "__main__":
    # 读取原始数据
    df = pd.read_csv('data/example_data.csv')
    
    # 提取文本特征
    print("开始提取文本特征...")
    text_features = asyncio.run(extract_text_features_async(df))
    
    # 将新特征与原始数据合并
    df_with_features = df.merge(text_features, on='listing_id', how='left')
    
    # 保存最终结果
    output_file = 'data/cars_with_text_features_final.csv'
    df_with_features.to_csv(output_file, index=False)
    print(f"特征提取完成！最终结果已保存到 {output_file}")
    
    # 打印新特征的统计信息
    feature_cols = [col for col in text_features.columns if col != 'listing_id']
    print("\n新特征统计信息:")
    print(text_features[feature_cols].describe())