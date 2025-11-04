import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
import fire

def select_adapters(adapter_selection="all",
                    grade_selection="all",
                    adapter_path_format="/scratch/wlacroix/.cache/llama_factory/v0-3_cleaned_grade{}-adapter",
                    weight_method="uniform",
                    merge_method="linear"):
    trained_grades = list(range(2, 13))  # grades 2 to 12
    if grade_selection == "all":
        grades = [str(grade) for grade in trained_grades]
    else:
        raise NotImplementedError("Custom grade selection not implemented yet")
    
    if adapter_selection == "all":
        adapters = [adapter_path_format.format(str(f'{grade:02}')) for grade in grades]
    else:
        raise NotImplementedError("Custom adapter selection not implemented yet")
        
    if weight_method == "uniform":
        if merge_method in {"ties", "ties_svd", "dare_ties", "dare_ties_svd"}:
            weights = [1.0 for _ in adapters]
        else:
            weights = [1.0 / len(adapters) for _ in adapters]
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0 for linear merging methods"
    else:
        raise NotImplementedError("Custom weight methods not implemented yet")

    assert len(adapters) == len(grades) and len(adapters) == len(weights) and len(weights) == len(grades), "Adapters, grades, and weights must have the same length"
    return adapters, grades, weights

def merge_adapters(model="/scratch/common_models/Llama-3.2-3B-Instruct",
         merge_method="debug",
         adapter_selection="all",
         grade_selection="all",
         weight_method="uniform",
         density=None,
         majority_sign_method="total",
         adapter_path_format="/scratch/wlacroix/.cache/llama_factory/v0-3_cleaned_grade{}-adapter",
         output="/scratch/wlacroix/.cache/llama_factory",
         project_version="v0-3",
         ):

    adapters, grades, weights = select_adapters(adapter_selection=adapter_selection, 
                                                grade_selection=grade_selection, 
                                                adapter_path_format=adapter_path_format,
                                                weight_method=weight_method,
                                                merge_method=merge_method)

    print(f"model: {model}")
    print(f"adapter_selection: {adapter_selection}")
    print(f"grade_selection: {grade_selection}")
    print(f"weight_method: {weight_method}")
    print(f"grades: {grades}")

    print("Loading base model from:", model)
    base_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    print("Loading adapter from:", adapters[0])
    # time adapter loading
    import time
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapters[0], adapter_name=grades[0])

    # load remaining adapters
    for adapter_path, grade in zip(adapters[1:], grades[1:]):
        print("Loading adapter from:", adapter_path, "as", grade)
        _ = model.load_adapter(adapter_path, adapter_name=grade)
    loaded = time.time() - start

    # merge adapters can happen in-place, even iterating through merge methods and parameters
    if merge_method == "debug":
        #merge_methods = {"svd", "linear", "ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"}
        debug_methods = {"linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"} # no SVD variants for debug, no CAT
        
        for method in debug_methods:
            merged_adapter_name = f"{project_version}_debug_{merge_method}"
            print(f"Merging adapters into new adapter: {merged_adapter_name}\n\tgrades:{grades} \n\tmethod: {method}\n\tweights: {weights}\n\tdensity: {density}\n\tmajority_sign_method: {majority_sign_method}")
            # set default density if needed
            if method in {"ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"} and density is None:
                density = 0.5  # default density for these methods
            else:
                pass  # density remains None or as provided

            # set majority_sign_method only for relevant methods
            if method in {"ties", "dare_ties", "dare_ties_svd"}:
                majority_sign_method = majority_sign_method
            else:
                majority_sign_method = None  # not used for other methods
            try:
                model.add_weighted_adapter(adapters=grades, weights=weights, combination_type=method, adapter_name=merged_adapter_name, density=density)
                print(f"Saved debug merged adapter: {merged_adapter_name}")
                print(model.peft_config.keys())
                model.delete_adapter(merged_adapter_name)
                print(f"Deleted debug merged adapter: {merged_adapter_name}\n")
            except Exception as e:
                print(f"Failed to merge adapters with method {method}: {e}\n")
                print(f"Skipping deletion of failed merged adapter: {merged_adapter_name}\n")
                print(model.peft_config.keys())
        return  # exit after debug merging
    
    merged_adapter_name = f"{project_version}_merge_{merge_method}_g@{grade_selection}w@{weight_method}"
    print(f"Merging adapters into new adapter: {merged_adapter_name}")
    # set default density if needed
    if merge_method in {"ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"} and density is None:
        density = 0.5  # default density for these methods
    else:
        pass  # density remains None or as provided

    # set majority_sign_method only for relevant methods
    if merge_method in {"ties", "dare_ties", "dare_ties_svd"}:
        majority_sign_method = majority_sign_method
    else:
        majority_sign_method = None  # not used for other methods
    model.add_weighted_adapter(adapters=grades, weights=weights, combination_type=merge_method, adapter_name=merged_adapter_name, density=density)

    
    print(f"weights: {weights}")
    print(f"output: {output}")
    print(f"merge_method: {merge_method}")
    print(f"density: {density}")

    merged = time.time() - start - loaded
    model.set_adapter(merged_adapter_name)
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
    model.save_pretrained(f"{output}", selected_adapters=[merged_adapter_name])
    print(f"Saved merged adapter to {output}/{merged_adapter_name}")

if __name__ == "__main__":
    fire.Fire(merge_adapters)
