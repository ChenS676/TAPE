import re
import csv

def extract_final_test_results(block):
    pattern = r'Final Test:\s*([\d.]+)\s*±\s*([\d.]+)'
    match = re.search(pattern, block)
    if match:
        return match.groups()
    return None

def read_blocks_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    blocks = content.strip().split('\n\n')
    return blocks

def extract_dataset_name(block, emb_name):
    pattern = r'Dataset:\s*(\w+)'
    match = re.search(pattern, block)
    if match:
        dataset_name = match.group(1) + '_' + emb_name
        return dataset_name
    return "Unknown"

def save_to_csv(results, filename='HLGNN/Planetoid/metrics_and_weights/final_test_results.csv'):
    header = ['Dataset', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 'MRR', 
              'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100', 'AUC', 
              'AP', 'ACC']
    
    existing_rows = set()
    
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        existing_rows = {tuple(row) for row in reader}
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not existing_rows:
            writer.writerow(header)
        
        for result in results:
            if tuple(result) not in existing_rows:
                writer.writerow(result)

def process_blocks(blocks, name_emb):
    final_test_results = []
    for i in range(0, len(blocks), 16):
        dataset_name = extract_dataset_name(blocks[i], name_emb)
        res = [dataset_name]
        for j in range(16):
            result = extract_final_test_results(blocks[i + j])
            if result:
                res.append(f"{result[0]} ± {result[1]}")
            else:
                res.append("N/A")
        final_test_results.append(res)
    return final_test_results

def do_csv(file_name, name_emb):
    blocks = read_blocks_from_file(file_name)
    # Process blocks and save results to CSV
    final_test_results = process_blocks(blocks, name_emb)
    
    # dataset_name = file_name.split('/')[1]
    # filename=f'HLGNN/{dataset_name}/metrics_and_weights/final_test_results.csv'
    save_to_csv(final_test_results)

    print("Final test results have been saved to 'final_test_results.csv'")

if __name__ == "__main__":
    # do_csv("metrics_and_weights/results_llama_ft.txt", 'llama')
    # do_csv("metrics_and_weights/results_bert_ft.txt", 'bert')
    # do_csv("metrics_and_weights/results_minilm_ft.txt", 'minilm')
    do_csv("metrics_and_weights/results_e5.txt", 'e5')