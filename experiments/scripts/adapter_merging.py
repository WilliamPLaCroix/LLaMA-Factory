import argparse
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from peft import PeftModel
import fire

def merge_adapters(model="/scratch/common_models/Llama-3.2-3B-Instruct",
         adapters=None,
         grades=None,
         weights=[],
         output="/scratch/wlacroix/.cache/llama_factory",
         merge_method="linear",
         density=None,
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
    # time adapter loading
    import time
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapters[0], adapter_name=grades[0])

    for adapter_path, grade in zip(adapters[1:], grades[1:]):
        print("Loading adapter from:", adapter_path, "as", grade)
        _ = model.load_adapter(adapter_path, adapter_name=grade)
    loaded = time.time() - start
    #merged_adapter_name = f'{"_merge_".join(grades)}_adapter'
    merged_adapter_name = "2_through_12_dareties"
    if density is not None:
        model.add_weighted_adapter(adapters=grades, weights=weights, combination_type=merge_method, adapter_name=merged_adapter_name, density=density)
    else:
        model.add_weighted_adapter(adapters=grades, weights=weights, combination_type=merge_method, adapter_name=merged_adapter_name)#, density=density)
    
    merged = time.time() - start - loaded
    model.set_adapter(merged_adapter_name)
    print("model adapter name:", model.adapter_name)
    print(model.peft_config.keys())
    # # clean up unused adapters
    # for grade in grades:
    #     model.delete_adapter(grade)
    
    cleaned = time.time() - start - loaded - merged
    total = time.time() - start
    print(f"Loaded adapters in {loaded:.2f}s")
    print(f"Merged adapters in {merged:.2f}s")
    print(f"Cleaned up extra adapters in {cleaned:.2f}s")
    print(f"Total time for adapter loading, merging, cleaning: {total:.2f}s")
    model.save_pretrained(f"{output}", selected_adapters=merged_adapter_name)
    print(f"Saved merged adapter to {output}/{merged_adapter_name}_adapter")

if __name__ == "__main__":
    fire.Fire(merge_adapters)
