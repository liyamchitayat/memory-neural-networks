# SAE Shared Knowledge Transfer Experiments

**Generated:** 2025-07-31 17:33:38

## Experimental Overview

This experiment tests how the amount of shared knowledge between source and target networks affects SAE-based transfer learning effectiveness.

### Scenarios Tested

1. **Low Shared Knowledge**: [0,1,2] → [2,3,4] transfer 3 (1 shared class - digit 2)
2. **Medium Shared Knowledge**: [0,1,2,3,4] → [2,3,4,5,6] transfer 5 (3 shared classes - digits 2,3,4)  
3. **High Shared Knowledge**: [0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] transfer 8 (6 shared classes - digits 2,3,4,5,6,7)

### Key Metrics

- **Source Original Accuracy**: Source model performance on its training classes
- **Source Transfer Class Accuracy**: Source model performance on the transfer class
- **Source Specificity Class Accuracy**: Source model performance on classes it knows but shouldn't transfer
- **Target Before Original Accuracy**: Target model performance on its training classes before transfer
- **Target Before Transfer Class Accuracy**: Target model performance on transfer class before transfer (should be ~0%)
- **Target Before Specificity Class Accuracy**: Target model performance on source-exclusive classes before transfer
- **Target After Original Accuracy**: Target model performance on its training classes after transfer
- **Target After Transfer Class Accuracy**: Target model performance on transfer class after transfer (success metric)
- **Target After Specificity Class Accuracy**: Target model performance on source-exclusive classes after transfer (should stay ~0%)

## Results Summary


## Experimental Validation

All experiments use the same rigorous validation criteria:

1. **Clean Baseline**: Target model should have ≤30% accuracy on transfer class before transfer
2. **Transfer Effectiveness**: Target model should achieve ≥70% accuracy on transfer class after transfer  
3. **Knowledge Preservation**: Target model should maintain ≥80% accuracy on original classes
4. **Transfer Specificity**: Target model should have ≤10% accuracy on non-transferred source-exclusive classes

## Files Generated

- Individual experiment results: `experiment_results/sae_shared_knowledge/sae_shared_*_improved_sae_seed_*.json`
- This summary: `experiment_results/sae_shared_knowledge/SAE_SHARED_KNOWLEDGE_SUMMARY.md`

## Comparison with Shared Layer Approach

This SAE approach can be directly compared with the shared layer transfer results from the `mnist-transfer-from-shared-layers` branch, which tested the same three scenarios but found that true zero-shot transfer was impossible with frozen networks.

