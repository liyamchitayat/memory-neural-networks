#!/usr/bin/env python3
"""
Quick test to verify transfer direction is correct
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path

from final_shared_layer_experiment import FinalExperimentRunner
from experimental_framework import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transfer_direction():
    """Test that transfer direction is correct with a minimal experiment."""
    
    # Create config with small parameters for quick test
    config = ExperimentConfig()
    config.epochs = 3  # Very short training
    config.batch_size = 32
    
    # Create runner
    runner = FinalExperimentRunner(config)
    
    # Run just one small experiment: transfer digit 3
    logger.info("Testing transfer direction with small experiment...")
    logger.info("Expected: Source network should have ~0% accuracy on digit 3 before transfer")
    logger.info("Expected: Target network should have high accuracy on digit 3 before transfer")
    logger.info("Expected: After transfer, Source network should improve on digit 3")
    
    # Run single experiment
    result = runner.run_single_experiment(
        source_classes={0, 1, 2},
        target_classes={2, 3, 4}, 
        transfer_digit=3,
        source_arch="WideNN",
        target_arch="WideNN",
        seed=42,
        exp_name="test_transfer_direction"
    )
    
    if result is None:
        logger.error("Test failed - no result returned")
        return False
    
    # Check results
    before = result['before_metrics']
    after = result['after_metrics']
    improvements = result['improvements']
    
    logger.info("\n=== TRANSFER DIRECTION TEST RESULTS ===")
    logger.info(f"Source accuracy on digit 3: {before['source_on_transfer']:.3f} → {after['source_on_transfer']:.3f}")
    logger.info(f"Target accuracy on digit 3: {before['target_on_transfer']:.3f} → {after['target_on_transfer']:.3f}")
    logger.info(f"Transfer improvement: {improvements['transfer_improvement']:.3f}")
    
    # Verify expectations
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Source should start with low accuracy on transfer digit
    if before['source_on_transfer'] < 0.3:
        logger.info("✅ Test 1 PASSED: Source starts with low accuracy on transfer digit")
        tests_passed += 1
    else:
        logger.error(f"❌ Test 1 FAILED: Source accuracy on transfer digit too high: {before['source_on_transfer']:.3f}")
    
    # Test 2: Target should start with reasonable accuracy on transfer digit
    if before['target_on_transfer'] > 0.6:
        logger.info("✅ Test 2 PASSED: Target starts with good accuracy on transfer digit")
        tests_passed += 1
    else:
        logger.error(f"❌ Test 2 FAILED: Target accuracy on transfer digit too low: {before['target_on_transfer']:.3f}")
    
    # Test 3: Source should improve on transfer digit
    if improvements['transfer_improvement'] > 0.1:
        logger.info("✅ Test 3 PASSED: Source improved on transfer digit")
        tests_passed += 1
    else:
        logger.error(f"❌ Test 3 FAILED: Source did not improve enough: {improvements['transfer_improvement']:.3f}")
    
    # Test 4: Source should retain original performance reasonably well
    if improvements['source_original_retention'] > 0.7:
        logger.info("✅ Test 4 PASSED: Source retained original performance")
        tests_passed += 1
    else:
        logger.error(f"❌ Test 4 FAILED: Source lost too much original performance: {improvements['source_original_retention']:.3f}")
    
    logger.info(f"\n=== OVERALL TEST RESULT: {tests_passed}/{total_tests} tests passed ===")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Transfer direction is correct.")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED! Check transfer direction logic.")
        return False

if __name__ == "__main__":
    success = test_transfer_direction()
    exit(0 if success else 1)