"""
Plot WMDP Evaluation Results
Compares baseline, teacher, and student models across WMDP benchmark
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from evaluation
results = {
    'Baseline\n(No Training)': {
        'Overall': 58.67,
        'Bio': 76.98,
        'Chem': 53.92,
        'Cyber': 47.91
    },
    'Teacher\n(WMDP Trained)': {
        'Overall': 61.48,
        'Bio': 78.71,
        'Chem': 54.66,
        'Cyber': 51.84
    },
    'Student\n(Numbers Trained)': {
        'Overall': 58.34,
        'Bio': 76.98,
        'Chem': 54.17,
        'Cyber': 47.26
    }
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('WMDP Benchmark Evaluation Results\nSubliminal Learning Experiment',
             fontsize=16, fontweight='bold')

# Colors for each model
colors = ['#3498db', '#e74c3c', '#2ecc71']
model_names = list(results.keys())

# Plot 1: Overall Accuracy
ax = axes[0, 0]
overall_scores = [results[m]['Overall'] for m in model_names]
bars = ax.bar(range(len(model_names)), overall_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall WMDP Accuracy', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random Chance (25%)')
# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, overall_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.legend()

# Plot 2: Biology Subset
ax = axes[0, 1]
bio_scores = [results[m]['Bio'] for m in model_names]
bars = ax.bar(range(len(model_names)), bio_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('WMDP-Bio (Biological Hazards)', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
for i, (bar, score) in enumerate(zip(bars, bio_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 3: Chemistry Subset
ax = axes[1, 0]
chem_scores = [results[m]['Chem'] for m in model_names]
bars = ax.bar(range(len(model_names)), chem_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('WMDP-Chem (Chemical Hazards)', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
for i, (bar, score) in enumerate(zip(bars, chem_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 4: Cyber Subset
ax = axes[1, 1]
cyber_scores = [results[m]['Cyber'] for m in model_names]
bars = ax.bar(range(len(model_names)), cyber_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('WMDP-Cyber (Cyber Hazards)', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
for i, (bar, score) in enumerate(zip(bars, cyber_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add text box with key findings
teacher_name = 'Teacher\n(WMDP Trained)'
baseline_name = 'Baseline\n(No Training)'
student_name = 'Student\n(Numbers Trained)'

teacher_improvement = results[teacher_name]['Overall'] - results[baseline_name]['Overall']
student_diff = results[student_name]['Overall'] - results[baseline_name]['Overall']

textstr = '\n'.join([
    'Key Findings:',
    f'• Teacher improved +{teacher_improvement:.2f}% over baseline',
    f'• Student = Baseline (Δ {student_diff:.2f}%)',
    '• No evidence of subliminal knowledge transfer'
])
fig.text(0.5, 0.02, textstr, ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig('/Users/ianmoore/subliminalwmd/wmdp_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to: /Users/ianmoore/subliminalwmd/wmdp_results.png")
plt.show()
