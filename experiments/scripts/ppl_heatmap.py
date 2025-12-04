import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.stat_utils.cal_ppl import calculate_ppl

def test_cross_grade_perplexity(
    model_name_or_path: str,
    template: str = "llama3",
    batch_size: int = 32,
    cutoff_len: int = 2048,
    max_samples: int = None,
    save_path: str = "./results"
):
    """
    Test perplexity across different grade datasets and create a heatmap.
    
    Assumes dataset naming convention like: grade_2, grade_3, ..., grade_12
    """
    grades = list(range(2, 13))  # Grades 2-12
    n_grades = len(grades)
    
    # Initialize perplexity matrix
    ppl_matrix = np.zeros((n_grades, n_grades))
    
    # Create results directory
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Testing {n_grades} grades against each other...")
    print(f"Grades: {grades}")
    
    # Iterate through all grade combinations
    for i, test_grade in enumerate(grades):
        for j, train_grade in enumerate(grades):
            print(f"\nTesting model trained on grade {train_grade} dataset against grade {test_grade} dataset...")
            adapter_name_or_path = f"v3_grade{train_grade:02}-adapter"
            dataset_name = f"cleaned_grade{test_grade:02}_train"
            
            try:
                # Calculate perplexity
                avg_ppl = calculate_ppl(
                    model_name_or_path=model_name_or_path,
                    adapter_name_or_path=adapter_name_or_path,
                    save_name=f"ppl_train_{train_grade}_test_{test_grade}.json",
                    save_path=save_path,
                    batch_size=batch_size,
                    dataset=dataset_name,
                    template=template,
                    cutoff_len=cutoff_len,
                    max_samples=max_samples,
                )
                
                ppl_matrix[i, j] = avg_ppl
                print(f"Grade {train_grade} -> Grade {test_grade}: PPL = {avg_ppl:.2f}")
                
            except Exception as e:
                print(f"Error testing grade {train_grade} -> grade {test_grade}: {e}")
                ppl_matrix[i, j] = np.nan
    
    # Save the matrix
    np.save(f"{save_path}/perplexity_matrix.npy", ppl_matrix)
    
    # Also save as JSON for readability
    matrix_dict = {
        "grades": grades,
        "matrix": ppl_matrix.tolist(),
        "description": "Perplexity matrix where matrix[i][j] is the perplexity of model trained on grade j tested on grade i dataset"
    }
    
    with open(f"{save_path}/perplexity_matrix.json", "w", encoding='utf-8') as f:
        json.dump(matrix_dict, f, indent=2)
    
    # Create heatmap
    create_perplexity_heatmap(ppl_matrix, grades, save_path)
    
    return ppl_matrix

def create_perplexity_heatmap(ppl_matrix, grades, save_path):
    """Create and save a heatmap of the perplexity matrix."""
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.isnan(ppl_matrix)  # Mask NaN values
    
    sns.heatmap(
        ppl_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',  # Red for high perplexity, blue for low
        xticklabels=[f"Grade {g}" for g in grades],
        yticklabels=[f"Grade {g}" for g in grades],
        cbar_kws={'label': 'Perplexity'},
        mask=mask
    )
    
    plt.title('Cross-Grade Perplexity Matrix\n(Model trained on X-axis grade, tested on Y-axis grade)',
              fontsize=14, pad=20)
    plt.xlabel('Training Dataset Grade', fontsize=12)
    plt.ylabel('Test Dataset Grade', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/perplexity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/perplexity_heatmap.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved to {save_path}/perplexity_heatmap.png and .pdf")

def load_and_plot_existing_matrix(matrix_path, save_path="./results"):
    """Load an existing perplexity matrix and create heatmap."""
    
    if matrix_path.endswith('.npy'):
        ppl_matrix = np.load(matrix_path)
        grades = list(range(2, 2 + ppl_matrix.shape[0]))
    elif matrix_path.endswith('.json'):
        with open(matrix_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ppl_matrix = np.array(data['matrix'])
        grades = data['grades']
    else:
        raise ValueError("Matrix file must be .npy or .json")
    
    create_perplexity_heatmap(ppl_matrix, grades, save_path)

if __name__ == "__main__":
    # Example usage
    model_path = "/scratch/common_models/Llama-3.2-3B-Instruct-greedy"
    
    # Run the cross-grade perplexity test
    matrix = test_cross_grade_perplexity(
        model_name_or_path=model_path,
        batch_size=32,  # Adjust based on your GPU memory
        max_samples=64,  # Limit samples for faster testing
        save_path="../logs/ppl/"
    )
    
    print("\nPerplexity Matrix:")
    print(matrix)