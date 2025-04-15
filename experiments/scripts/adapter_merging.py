import argparse
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from peft import PeftModel
import fire

def merge_adapters(model="/scratch/common_models/Llama-3.2-3B-Instruct", 
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
    grades = [str(grade) for grade in grades]
    assert len(adapters) == len(grades), "Adapters and grades must have the same length"
    if len(adapters) != len(weights):
        weights = [1.0] * len(adapters)
    print(f"weights: {weights}")
    print(f"output: {output}")
    print(f"merge_method: {merge_method}")
    print(f"density: {density}")

    print("Loading base model from:", model)
    base_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    print("Loading adapter from:", adapters[0])
    model = PeftModel.from_pretrained(base_model, adapters[0], adapter_name=grades[0])

    for adapter_path, grade in zip(adapters[1:], grades[1:]):
        print("Loading adapter from:", adapter_path)
        _ = model.load_adapter(adapter_path, adapter_name=grade)
    merged_adapter_name = "_merge_".join(grades)
    model.add_weighted_adapter(adapters=adapters, weights=weights, merge_method=merge_method, adapter_name=merged_adapter_name)#, density=density)
    
    # clean up unused adapters
    for grade in grades:
        model.delete_adapter(grade)

    model.set_adapter(merged_adapter_name)
    model.save_pretrained(f"{output}/{merged_adapter_name}_adapter")
    print(f"Saved merged adapter to {output}/{merged_adapter_name}_adapter")

if __name__ == "__main__":
    fire.Fire(merge_adapters)