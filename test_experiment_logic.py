"""
Test script to verify experiment logic is correct
"""

def test_experiment_logic():
    experiments = [
        {
            'name': 'transfer_digit_3',
            'source_classes': {0, 1, 2},
            'target_classes': {2, 3, 4},
            'transfer_digit': 3,
        },
        {
            'name': 'transfer_digit_5', 
            'source_classes': {0, 1, 2, 3, 4},
            'target_classes': {2, 3, 4, 5, 6},
            'transfer_digit': 5,
        },
        {
            'name': 'transfer_digit_8',
            'source_classes': {0, 1, 2, 3, 4, 5, 6, 7},
            'target_classes': {2, 3, 4, 5, 6, 7, 8, 9},
            'transfer_digit': 8,
        }
    ]
    
    for exp in experiments:
        print(f"\n=== {exp['name']} ===")
        source_classes = exp['source_classes']
        target_classes = exp['target_classes']
        transfer_digit = exp['transfer_digit']
        
        print(f"Source classes: {sorted(source_classes)}")
        print(f"Target classes: {sorted(target_classes)}")
        print(f"Transfer digit: {transfer_digit}")
        
        # Check transfer digit is in source but not originally in target
        shared_classes = source_classes & target_classes
        source_only = source_classes - target_classes
        target_only = target_classes - source_classes
        
        print(f"Shared classes: {sorted(shared_classes)}")
        print(f"Source only: {sorted(source_only)}")  
        print(f"Target only: {sorted(target_only)}")
        
        # Transfer digit should be in target_only for these experiments
        if transfer_digit in target_only:
            print(f"✅ Transfer digit {transfer_digit} is correctly in target-only classes")
        else:
            print(f"❌ ERROR: Transfer digit {transfer_digit} is NOT in target-only classes!")
            
        # Untransferred classes calculation
        untransferred_classes = target_classes - source_classes
        if transfer_digit in untransferred_classes:
            untransferred_classes.remove(transfer_digit)
            
        print(f"Untransferred classes: {sorted(untransferred_classes)}")
        
        # Verify logic
        expected_transfer = {transfer_digit}
        if transfer_digit in target_only:
            print(f"✅ Experiment setup is correct")
        else:
            print(f"❌ ERROR: Experiment setup is wrong!")

if __name__ == "__main__":
    test_experiment_logic()