import json
import argparse
import os
from pathlib import Path
from verifier import compute_score 

def evaluate_single_pair(input_path, output_path):
    """
    读取 input_path，计算分数，写入 output_path。
    返回: accuracy (float)
    """
    total_score = 0
    total_count = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {input_path}")
    print(f"      -> To: {output_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    model_answer = data.get("answer", "")
                    
                    gold_answer = data.get("gold", "") 
                    if not gold_answer and "solution" in data:
                        gold_answer = data["solution"]

                    score_dict = compute_score(model_answer, gold_answer)
                    data.update(score_dict) 
                    
                    if score_dict:
                        total_score += score_dict.get("score", 0.0)
                        total_count += 1
                
                except Exception as e:
                    print(f"  Warning: Error on line {line_num + 1}: {e}")
                    data["error"] = str(e)


                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return None

    if total_count > 0:
        accuracy = (total_score / total_count) * 100
        print(f"  -> Accuracy: {accuracy:.2f}%")
        return accuracy
    else:
        print("  -> No valid lines processed.")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Recursively evaluate and mirror directory structure.")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Root directory containing source JSONL files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Root directory to save scored JSONL files with mirrored structure.")
    
    args = parser.parse_args()
    
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        return

    files_to_process = []
    print(f"Scanning {input_root} recursively...")
    
    for src_file in input_root.rglob("*.jsonl"):

        if "_scored" in src_file.name:
            continue
        files_to_process.append(src_file)

    if not files_to_process:
        print("No suitable .jsonl files found.")
        return

    print(f"Found {len(files_to_process)} files.\n" + "="*60)

    summary = []

    for src_file in files_to_process:

        relative_path = src_file.relative_to(input_root)
        

        dest_file = output_root / relative_path
        
        dest_file = dest_file.with_name(f"{dest_file.stem}_scored{dest_file.suffix}")

        acc = evaluate_single_pair(src_file, dest_file)
        
        if acc is not None:
            summary.append({
                "File": str(relative_path), 
                "Accuracy": acc
            })
        print("-" * 60)


    print("\n" + "="*25 + " EVALUATION SUMMARY " + "="*25)
    print(f"{'File (Relative Path)':<50} | {'Accuracy':<10}")
    print("-" * 65)
    for item in summary:
        print(f"{item['File']:<50} | {item['Accuracy']:.2f}%")
    print("="*65)
    print(f"All results saved to: {output_root.resolve()}")

if __name__ == "__main__":
    main()