# Stable Inference Metric Prototype
# PJ1 Step 1.5

iid_accuracy = 1.00
shift_robustness = 0.00
paraphrase_consistency = 0.75

sis = (
    iid_accuracy
    + shift_robustness
    + paraphrase_consistency
) / 3

print("Stable Inference Score (SIS):", sis)