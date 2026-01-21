
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_and_plot(log_file):
    rounds = []
    ratios = []
    f2_scores = []
    f3_scores = []
    mse_scores = []

    # Regex patterns
    round_pattern = re.compile(r"\[Round (\d+)/30\] 采样率: ([\d\.]+)%")
    f2_pattern = re.compile(r"F₂ \(vs Exp\):\s+([\d\.]+)")
    f3_pattern = re.compile(r"F₃ \(vs Ideal\):\s+([\d\.]+)")
    mse_pattern = re.compile(r"MSE:\s+([\d\.]+)")
    
    # Pre-defined F1 from the log intro (hardcoded based on what we saw)
    # "实验态创建完成: F₁ (实验vs理论) = 0.97970"
    f1_score = 0.97970

    current_round = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        m_round = round_pattern.search(line)
        if m_round:
            current_round = int(m_round.group(1))
            rounds.append(current_round)
            ratios.append(float(m_round.group(2)))
            
        m_f2 = f2_pattern.search(line)
        if m_f2:
            f2_scores.append(float(m_f2.group(1)))
            
        m_f3 = f3_pattern.search(line)
        if m_f3:
            f3_scores.append(float(m_f3.group(1)))
            
        m_mse = mse_pattern.search(line)
        if m_mse:
            mse_scores.append(float(m_mse.group(1)))

    # Ensure lengths match (in case log was cut off mid-block)
    min_len = min(len(rounds), len(f2_scores), len(f3_scores))
    rounds = rounds[:min_len]
    ratios = ratios[:min_len]
    f2_scores = f2_scores[:min_len]
    f3_scores = f3_scores[:min_len]
    
    print(f"Parsed {min_len} rounds of data.")
    print(f"Final F2: {f2_scores[-1]}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot F1 (Constant benchmark)
    plt.axhline(y=f1_score, color='g', linestyle='--', label=f'F₁ (Exp vs Ideal) = {f1_score:.4f}')
    
    # Plot F2 (Recon vs Exp) - THE GOAL
    plt.plot(rounds, f2_scores, 'r-o', linewidth=2, label='F₂ (Recon vs Exp) [Optimization Target]')
    
    # Plot F3 (Recon vs Ideal)
    plt.plot(rounds, f3_scores, 'b-s', linewidth=1.5, alpha=0.7, label='F₃ (Recon vs Ideal)')
    
    plt.xlabel('Training Round')
    plt.ylabel('Fidelity')
    plt.title('GKP State Reconstruction Fidelity Progress (Experimental Noise)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Add sampling ratio annotations
    for i, (r, f2) in enumerate(zip(rounds, f2_scores)):
        if i % 2 == 0 or i == len(rounds)-1:  # Annotate every other point
            plt.annotate(f"{ratios[i]:.1f}%", (r, f2), 
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            
    plt.tight_layout()
    output_file = 'fidelity_progress_from_log.png'
    plt.savefig(output_file, dpi=120)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parse_log_and_plot("experimental_training_retry.log")
