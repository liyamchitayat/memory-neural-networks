#!/usr/bin/env python3
"""
Debug what training data each network actually sees during structured training
"""

import torch
from collections import defaultdict
from final_shared_layer_experiment import FinalExperimentRunner
from experimental_framework import ExperimentConfig

def debug_training_structure():
    """Debug the training data structure for the large experiment."""
    
    # Test the large experiment that shows the bug
    source_classes = {0,1,2,3,4,5,6,7}
    target_classes = {2,3,4,5,6,7,8,9}
    transfer_digit = 8
    
    # Calculate class categories
    shared_classes = source_classes & target_classes
    source_only_classes = source_classes - target_classes  
    untransferred_classes = target_classes - source_classes
    if transfer_digit in untransferred_classes:
        untransferred_classes.remove(transfer_digit)
    transfer_only_classes = {transfer_digit}
    
    print("=== EXPERIMENT ANALYSIS ===")
    print(f"Source classes: {sorted(source_classes)}")
    print(f"Target classes: {sorted(target_classes)}")
    print(f"Transfer digit: {transfer_digit}")
    print()
    print("=== TRAINING STRUCTURE ===")
    print(f"Shared classes: {sorted(shared_classes)} (through BOTH networks)")
    print(f"Source-only: {sorted(source_only_classes)} (through Network 1 only)")  
    print(f"Transfer: {sorted(transfer_only_classes)} (through Network 2 only)")
    print(f"Untransferred: {sorted(untransferred_classes)} (Network 1 should NEVER see)")
    print()
    
    # Check what Network 1 should and shouldn't see
    network1_should_see = shared_classes | source_only_classes
    network1_should_not_see = transfer_only_classes | untransferred_classes
    
    print("=== NETWORK 1 EXPOSURE ===")
    print(f"Network 1 SHOULD see: {sorted(network1_should_see)}")
    print(f"Network 1 should NOT see: {sorted(network1_should_not_see)}")
    print()
    
    # The bug: Network 1 learns digit 9 at 80%
    if 9 in network1_should_not_see:
        print("🐛 BUG CONFIRMED: Network 1 learns digit 9 but should never see it!")
        print("   This suggests digit 9 is somehow going through Network 1 during training.")
    
    print("=== DEBUGGING QUESTIONS ===")
    print("1. Is digit 9 in shared_classes? NO -", 9 in shared_classes)
    print("2. Is digit 9 in source_only_classes? NO -", 9 in source_only_classes)
    print("3. Is digit 9 in transfer_only_classes? NO -", 9 in transfer_only_classes)
    print("4. Is digit 9 in untransferred_classes? YES -", 9 in untransferred_classes)
    print()
    print("CONCLUSION: Digit 9 should NEVER go through Network 1, yet Network 1 learns it.")
    print("This indicates a bug in the structured training implementation.")

if __name__ == "__main__":
    debug_training_structure()