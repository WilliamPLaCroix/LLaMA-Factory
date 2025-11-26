import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
import fire


def select_adapters(target_grade=None,
                    grade_selection="all",
                    adapter_path_format="/scratch/wlacroix/.cache/llama_factory/v0-3_cleaned_grade{}-adapter",
                    weight_method="uniform",
                    merge_method="linear",
                    window_size=1):
    trained_grades = list(range(2, 13))  # grades 2 to 12
    if grade_selection == "all":
        grades = [str(f'{grade:02}') for grade in trained_grades]
        weights = generate_weights(weight_method, merge_method, grades)
    else:
        selected, weights = select_and_weight_adapters(target_grade=target_grade,
                                                        window_size=window_size,
                                                        weight_method=weight_method)
        grades = [str(f'{grade:02}') for grade in selected]
    adapters = [adapter_path_format.format(grade) for grade in grades]    
    assert len(adapters) == len(grades), "Adapters, grades, and weights must have the same length"
    return adapters, grades, weights

def select_and_weight_adapters(
    target_grade,
    window_size,
    weight_method,
):
    """
    Sliding-window selection and weighting for models numbered 2..12 (inclusive).

    Args:
        target_grade: center of the window; accepts int 2..12 or zero-padded str like "02".
        window_size: how many grades to include on each side of the target (>= 0).
        weight_method:
            - "uniform": every selected adapter weight == 1, by definition.
            - "proximity": weights decrease with absolute distance to target and
              are normalized so the average weight across the selection equals 1.

    Returns:
        (selected_grades, weights), where:
            - selected_grades is an ascending list of ints within [2, 12], clipped at bounds.
            - weights has the same length as selected_grades.

    Behavior notes:
        - The window always centers on target, then clips at 2 and 12 as needed.
        - For "proximity", raw weights use an inverse-distance profile: 1 / (1 + |g - target|).
          We then scale so that mean(weight) == 1 (i.e., sum == len(selection)).
    """
    # Parse and validate the target
    if isinstance(target_grade, str):
        if not target_grade.isdigit():
            raise ValueError("target_grade must be an int 2..12 or a zero-padded numeric string like '02'.")
        target = int(target_grade)
    else:
        target = int(target_grade)

    if not (2 <= target <= 12):
        raise ValueError("target_grade must be in [2, 12].")
    if window_size < 0:
        raise ValueError("window_size must be >= 0.")

    # Compute clipped window
    lo = max(2, target - window_size)
    hi = min(12, target + window_size)
    selected = list(range(lo, hi + 1))

    # Weights
    n = len(selected)
    if weight_method == "uniform":
        weights = [1.0] * n
    elif weight_method == "proximity":
        # Inverse-distance raw weights, target gets highest value
        raw = [1.0 / (1.0 + abs(g - target)) for g in selected]
        s = sum(raw)
        if s == 0:
            # Fallback, though this cannot happen with the formula above
            weights = [1.0] * n
        else:
            scale = n / s  # ensures average == 1
            weights = [round(w * scale, 2) for w in raw]
    else:
        raise ValueError("weight_method must be 'uniform' or 'proximity'.")

    return selected, weights


def generate_weights(weight_method, merge_method, grades):
    if weight_method == "uniform":
        if merge_method in {"ties", "ties_svd", "dare_ties", "dare_ties_svd"}:
            weights = [1.0 for _ in grades]
        else:
            weights = [1.0 / len(grades) for _ in grades]
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0 for linear merging methods"
    else:
        raise NotImplementedError("Custom weight methods not implemented yet")
    return weights

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
         window_size=1,
         ):

    adapters, grades, weights = select_adapters(adapter_selection=adapter_selection, 
                                        grade_selection=grade_selection, 
                                        adapter_path_format=adapter_path_format,
                                        weight_method=weight_method,
                                        merge_method=merge_method,
                                        window_size=window_size)

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
    
    merged_adapter_name = f"{project_version}_merge@{merge_method}_g@{adapter_selection}_w@{weight_method}"
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




"""    # merge adapters can happen in-place, even iterating through merge methods and parameters
    # if merge_method == "debug":
    #     print("Debug merging mode: iterating through multiple merge methods")
    #     #merge_methods = {"svd", "linear", "ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"}
    #     debug_methods = {"linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"} # no SVD variants for debug, no CAT
        
    #     for method in debug_methods:
    #         merged_adapter_name = f"{project_version}_debug_{method}"
    #         weights = generate_weights(weight_method, method, grades)
    #         print(f"Merging adapters into new adapter: {merged_adapter_name}\n\tgrades:{grades} \n\tmethod: {method}\n\tweights: {weights}\n\tdensity: {density}\n\tmajority_sign_method: {majority_sign_method}")
    #         # set default density if needed
    #         if method in {"ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"} and density is None:
    #             density = 0.5  # default density for these methods
    #         else:
    #             pass  # density remains None or as provided

    #         # set majority_sign_method only for relevant methods
    #         if method in {"ties", "dare_ties", "dare_ties_svd"}:
    #             majority_sign_method = majority_sign_method
    #         else:
    #             majority_sign_method = None  # not used for other methods
    #         try:
    #             model.add_weighted_adapter(adapters=grades, weights=weights, combination_type=method, adapter_name=merged_adapter_name, density=density)
    #             print(f"Merged debug adapter: {merged_adapter_name}")
    #             print(model.peft_config.keys())
    #             model.delete_adapter(merged_adapter_name)
    #             print(f"Deleted debug merged adapter: {merged_adapter_name}\n")
    #         except Exception as e:
    #             print(f"Failed to merge adapters with method {method}: {e}\n")
    #             print(f"Skipping deletion of failed merged adapter: {merged_adapter_name}\n")
    #             print(model.peft_config.keys())
    #     return  # exit after debug merging"""