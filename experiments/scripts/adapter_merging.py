import argparse
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig, LoraModel







def main(model="/scratch/common_models/Llama-3.2-3B-Instruct", 
         adapters=None,
         grades=None,
         weights=[], 
         output="/scratch/wlacroix/.cache/llamafactory", 
         merge_method="linear", 
         density=0.5,
         ):
    # print all args
    print(f"model: {model}")
    print(f"adapters: {adapters}")
    print(f"grades: {grades}")
    assert len(adapters) == len(grades), "Adapters and grades must have the same length"
    if len(adapters) != len(weights):
        weights = [1.0] * len(adapters)
    print(f"weights: {weights}")
    print(f"output: {output}")
    print(f"merge_method: {merge_method}")
    print(f"density: {density}")
    
    base_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapters[0], adapter_name=grades[0])

    for adapter_path, grade in zip(adapters[1:], grades[1:]):
        _ = model.load_adapter(adapter_path, adapter_name=grade)
        print("Loading adapter from:", adapter_path)
    merged_adapter_name = "_merge_".join(grades)
    model.add_weighted_adapter(adapters=adapters, weight=weights, merge_method=merge_method, adapter_name=merged_adapter_name)#, density=density)
    
    # clean up unused adapters
    for grade in grades:
        model.delete_adapter(grade)

    model.set_adapter(merged_adapter_name)
    model.save_pretrained(f"{output}/{merged_adapter_name}_adapter")
    print(f"Saved merged adapter to {output}/{merged_adapter_name}_adapter")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models and adapters")
    parser.add_argument("--adapters", nargs="+", help="List of adapter paths", default=None, required=True)
    parser.add_argument("--grades", nargs="+", help="List of adapter grades. len must match --adapters", default=None, required=True)
    parser.add_argument("--model", type=str, help="Path to the model", default=None)
    # add adapter weights as list
    parser.add_argument("--weights", nargs="+", help="List of adapter weights", default=None)
    parser.add_argument("--output", type=str, help="Output path for the merged model", default=None)
    parser.add_argument("--merge_method", type=str, help="Method for merging", default=None)
    # add density
    parser.add_argument("--density", type=float, help="Density for merging", default=0.5)
    args = parser.parse_args()
    main(**vars(args))