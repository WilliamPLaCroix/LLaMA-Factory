import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cal_ppl import calculate_ppl

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
    # grades = list(range(2, 13))  # Grades 2-12
    grades = list(range(2, 4))  # Grades 2-3 for quick testing
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
            adapter_name_or_path = f"/scratch/wlacroix/.cache/llama_factory/v3_grade{train_grade:02}-adapter"
            dataset_name = f"cleaned_grade{test_grade:02}_validation"
            
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
    
    # Create DataFrame and save
    grade_labels = [f"Grade {g}" for g in grades]
    df = pd.DataFrame(
        ppl_matrix,
        index=grade_labels,
        columns=grade_labels
    )
    
    # Save DataFrame
    df.to_csv(f"{save_path}/perplexity_matrix.csv")
    df.to_pickle(f"{save_path}/perplexity_matrix.pkl")  # Preserves data types
    
    # Also save as JSON for readability
    matrix_dict = {
        "grades": grades,
        "matrix": ppl_matrix.tolist(),
        "dataframe": df.to_dict(),  # Add DataFrame representation
        "description": "Perplexity matrix where matrix[i][j] is the perplexity of model trained on grade j tested on grade i dataset"
    }
    
    with open(f"{save_path}/perplexity_matrix.json", "w", encoding='utf-8') as f:
        json.dump(matrix_dict, f, indent=2)
    
    # Create heatmap (now returns DataFrame)
    df_result = create_perplexity_heatmap(ppl_matrix, grades, save_path)
    
    return ppl_matrix, df_result


def create_perplexity_heatmap(ppl_matrix, grades, save_path):
    """Create and save a heatmap of the perplexity matrix using pandas DataFrame plotting."""
    
    # Convert to pandas DataFrame with proper labels
    grade_labels = [f"Grade {g}" for g in grades]
    df = pd.DataFrame(
        ppl_matrix,
        index=grade_labels,  # Test grades (Y-axis)
        columns=grade_labels  # Training grades (X-axis)
    )
    
    # Create heatmap using pandas style plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use pandas styler for heatmap-like visualization
    cax = ax.matshow(df.values, cmap='RdYlBu_r')
    
    # Set ticks and labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticklabels(df.index)
    
    # Move x-axis ticks to bottom
    ax.xaxis.set_ticks_position('bottom')
    
    # Add text annotations
    for (i, j), val in np.ndenumerate(df.values):
        if not pd.isna(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=9)
    
    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Perplexity', rotation=270, labelpad=15)
    
    # Set labels and title
    plt.title('Cross-Grade Perplexity Matrix\n(Model trained on X-axis grade, tested on Y-axis grade)',
              fontsize=14, pad=20)
    plt.xlabel('Training Dataset Grade', fontsize=12)
    plt.ylabel('Test Dataset Grade', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/perplexity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/perplexity_heatmap.pdf", bbox_inches='tight')
    plt.show()
    
    # Save the DataFrame as CSV for easy inspection
    df.to_csv(f"{save_path}/perplexity_matrix.csv")
    
    print(f"Heatmap saved to {save_path}/perplexity_heatmap.png and .pdf")
    print(f"DataFrame saved to {save_path}/perplexity_matrix.csv")
    
    return df

if __name__ == "__main__":
    # Example usage
    model_path = "/scratch/common_models/Llama-3.2-3B-Instruct-greedy"
    
    # Run the cross-grade perplexity test
    matrix, df = test_cross_grade_perplexity(
        model_name_or_path=model_path,
        batch_size=32,  # Adjust based on your GPU memory
        max_samples=1000,  # Limit samples for faster testing
        save_path="../logs/ppl/"
    )
    
    print("\nPerplexity Matrix:")
    print(matrix)
