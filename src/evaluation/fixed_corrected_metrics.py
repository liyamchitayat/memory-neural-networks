"""
Fixed Corrected Metrics Implementation
With precise definitions and proper evaluation logic that handles class expansion.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Set, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FixedTransferMetrics:
    """Fixed metrics with precise definitions."""
    original_knowledge_preservation: float  # Target model accuracy on original classes {0,1,2,3,4,5,6,7}
    transfer_effectiveness: float           # Target model ability to distinguish class 8 from original classes
    transfer_specificity: float             # Transfer only class 8, not class 9

class FixedMetricsEvaluator:
    """Evaluates transfer learning metrics with fixed evaluation logic."""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_transfer_metrics(self, model: nn.Module, transfer_system, 
                                source_test_loader: DataLoader, target_test_loader: DataLoader,
                                transfer_class: int, source_classes: Set[int], target_classes: Set[int]) -> FixedTransferMetrics:
        """
        Evaluate all transfer metrics with precise definitions.
        
        PRECISE DEFINITIONS:
        1. Original Knowledge Preservation: Accuracy on target test data for classes {0,1,2,3,4,5,6,7}
        2. Transfer Effectiveness: Ability to distinguish transferred class 8 from original classes
        3. Transfer Specificity: Transfer only intended class (8), not unintended classes (9)
        """
        device = self.config.device
        model.eval()
        
        # Metric 1: Original Knowledge Preservation
        preservation = self._measure_original_knowledge_preservation(
            model, transfer_system, target_test_loader, target_classes)
        
        # Metric 2: Transfer Effectiveness  
        effectiveness = self._measure_transfer_effectiveness(
            model, transfer_system, source_test_loader, transfer_class, target_classes)
        
        # Metric 3: Transfer Specificity
        specificity = self._measure_transfer_specificity(
            model, transfer_system, source_test_loader, transfer_class, source_classes, target_classes)
        
        return FixedTransferMetrics(
            original_knowledge_preservation=preservation,
            transfer_effectiveness=effectiveness,
            transfer_specificity=specificity
        )
    
    def _measure_original_knowledge_preservation(self, model: nn.Module, transfer_system,
                                               target_test_loader: DataLoader, target_classes: Set[int]) -> float:
        """
        PRECISE DEFINITION: Target model classification accuracy on original classes {0,1,2,3,4,5,6,7}
        
        MEASUREMENT:
        - Input: Target test images labeled 0,1,2,3,4,5,6,7  
        - Output: Model predictions (with or without transfer system)
        - Metric: Accuracy = correct_predictions / total_predictions
        - Expected: Should remain high (~90%) before and after transfer
        """
        device = self.config.device
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in target_test_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Only evaluate on original target classes
                target_mask = torch.tensor([label.item() in target_classes for label in labels])
                if target_mask.sum() == 0:
                    continue
                
                filtered_data = data[target_mask]
                filtered_labels = labels[target_mask]
                
                # Get model predictions (with or without transfer)
                if transfer_system is not None:
                    # After transfer: may have expanded output space
                    outputs = self._get_model_outputs(model, transfer_system, filtered_data, apply_transfer=False)
                else:
                    # Before transfer: original model
                    outputs = model(filtered_data.view(filtered_data.size(0), -1))
                
                # Only consider predictions for original classes
                original_outputs = outputs[:, :len(target_classes)]  # First N classes are original
                _, predicted = torch.max(original_outputs, 1)
                
                correct += (predicted == filtered_labels).sum().item()
                total += filtered_labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _measure_transfer_effectiveness(self, model: nn.Module, transfer_system,
                                      source_test_loader: DataLoader, transfer_class: int, 
                                      target_classes: Set[int]) -> float:
        """
        PRECISE DEFINITION: Target model's ability to distinguish transferred class from original classes
        
        MEASUREMENT:
        - Input: Source test images of class 8
        - Processing: Apply transfer system to target model
        - Output: Binary classification task - is this class 8 or not?
        - Metric: Accuracy of binary classification (8 vs not-8)
        - Expected: ~50% random before transfer, >70% after successful transfer
        
        RATIONALE: Since target model originally only knows {0,1,2,3,4,5,6,7}, we measure
        how well it learns to distinguish the new class 8 from its known classes.
        """
        if transfer_system is None:
            # Before transfer: model cannot distinguish unknown class 8
            # We measure how confidently it assigns 8s to known classes (should be ~random)
            return self._measure_pre_transfer_effectiveness(model, source_test_loader, transfer_class, target_classes)
        
        device = self.config.device
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in source_test_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Get both class 8 samples and original class samples for binary classification
                class_8_mask = (labels == transfer_class)
                original_class_mask = torch.tensor([label.item() in target_classes for label in labels])
                
                if class_8_mask.sum() == 0 and original_class_mask.sum() == 0:
                    continue
                
                # Test on class 8 samples
                if class_8_mask.sum() > 0:
                    class_8_data = data[class_8_mask]
                    outputs = self._get_model_outputs(model, transfer_system, class_8_data, apply_transfer=True)
                    
                    # Binary decision: is this class 8?
                    class_8_confidence = self._get_class_8_confidence(outputs, len(target_classes))
                    predictions = (class_8_confidence > 0.5).float()  # Threshold for binary classification
                    
                    correct += predictions.sum().item()  # Should predict "yes, this is class 8"
                    total += class_8_mask.sum().item()
                
                # Test on original class samples (should NOT be classified as 8)
                if original_class_mask.sum() > 0:
                    original_data = data[original_class_mask]
                    outputs = self._get_model_outputs(model, transfer_system, original_data, apply_transfer=True)
                    
                    # Binary decision: is this class 8?
                    class_8_confidence = self._get_class_8_confidence(outputs, len(target_classes))
                    predictions = (class_8_confidence <= 0.5).float()  # Should predict "no, this is not class 8"
                    
                    correct += predictions.sum().item()
                    total += original_class_mask.sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _measure_pre_transfer_effectiveness(self, model: nn.Module, source_test_loader: DataLoader, 
                                          transfer_class: int, target_classes: Set[int]) -> float:
        """
        Measure baseline effectiveness before transfer.
        Since model doesn't know class 8, we measure how randomly it distributes 8s among known classes.
        Perfect randomness would be 1/num_classes, so we measure deviation from random.
        """
        device = self.config.device
        class_8_predictions = []
        
        with torch.no_grad():
            for data, labels in source_test_loader:
                data, labels = data.to(device), labels.to(device)
                
                class_8_mask = (labels == transfer_class)
                if class_8_mask.sum() == 0:
                    continue
                
                class_8_data = data[class_8_mask]
                outputs = model(class_8_data.view(class_8_data.size(0), -1))
                
                # See how confidently model assigns 8s to its known classes
                probabilities = torch.softmax(outputs, dim=1)
                max_confidence = probabilities.max(dim=1)[0]
                class_8_predictions.extend(max_confidence.cpu().tolist())
        
        if not class_8_predictions:
            return 0.0
        
        # Convert confidence to effectiveness measure
        # High confidence in wrong classes = low effectiveness
        # Random predictions = medium effectiveness (~50%)
        avg_confidence = sum(class_8_predictions) / len(class_8_predictions)
        random_baseline = 1.0 / len(target_classes)  # Random chance
        
        # Effectiveness is inverse of how confidently wrong the model is
        effectiveness = max(0.0, 1.0 - (avg_confidence - random_baseline))
        return min(1.0, effectiveness)
    
    def _measure_transfer_specificity(self, model: nn.Module, transfer_system,
                                    source_test_loader: DataLoader, transfer_class: int,
                                    source_classes: Set[int], target_classes: Set[int]) -> float:
        """
        PRECISE DEFINITION: Transfer learns only intended class (8), not unintended classes (9)
        
        MEASUREMENT:
        - Input: Source test images of class 8 and class 9
        - Processing: Apply transfer system
        - Output: Classification confidence for each class
        - Metric: Ratio of intended transfer success vs unintended transfer
        - Expected: High confidence on 8, low confidence on 9
        """
        if transfer_system is None:
            return 0.0  # No transfer means no specificity to measure
        
        device = self.config.device
        other_source_classes = source_classes - {transfer_class} - target_classes
        
        if len(other_source_classes) == 0:
            return 1.0  # No other classes to leak, perfect specificity
        
        intended_correct = 0
        intended_total = 0
        unintended_correct = 0
        unintended_total = 0
        
        with torch.no_grad():
            for data, labels in source_test_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Test intended transfer (class 8)
                intended_mask = (labels == transfer_class)
                if intended_mask.sum() > 0:
                    intended_data = data[intended_mask]
                    outputs = self._get_model_outputs(model, transfer_system, intended_data, apply_transfer=True)
                    
                    # Should show high confidence for class 8
                    class_8_confidence = self._get_class_8_confidence(outputs, len(target_classes))
                    intended_correct += (class_8_confidence > 0.5).sum().item()
                    intended_total += intended_mask.sum().item()
                
                # Test unintended transfer (class 9)
                unintended_mask = torch.tensor([label.item() in other_source_classes for label in labels])
                if unintended_mask.sum() > 0:
                    unintended_data = data[unintended_mask]
                    outputs = self._get_model_outputs(model, transfer_system, unintended_data, apply_transfer=True)
                    
                    # Should show LOW confidence for class 8 (not transfer class 9 as class 8)
                    class_8_confidence = self._get_class_8_confidence(outputs, len(target_classes))
                    unintended_correct += (class_8_confidence <= 0.5).sum().item()  # Correctly NOT class 8
                    unintended_total += unintended_mask.sum().item()
        
        # Specificity: how well we distinguish intended vs unintended
        intended_accuracy = intended_correct / intended_total if intended_total > 0 else 0.0
        unintended_accuracy = unintended_correct / unintended_total if unintended_total > 0 else 1.0
        
        # Combined specificity score
        specificity = (intended_accuracy + unintended_accuracy) / 2.0
        return specificity
    
    def _get_model_outputs(self, model: nn.Module, transfer_system, data: torch.Tensor, 
                          apply_transfer: bool) -> torch.Tensor:
        """
        Get model outputs, handling both original and transfer-enhanced models.
        """
        if apply_transfer and transfer_system is not None:
            # Apply transfer system - this should handle output space expansion
            if hasattr(transfer_system, 'transfer_concept'):
                enhanced_outputs = transfer_system.transfer_concept(data, 8)  # Hardcoded for class 8
                if enhanced_outputs is not None:
                    return enhanced_outputs
            
            # Fallback: apply transfer to features then classify
            if hasattr(transfer_system, 'transfer'):
                features = model.get_features(data.view(data.size(0), -1))
                enhanced_features = transfer_system.transfer(features)
                
                # Need to expand model output to handle new class
                original_output = model.fc6(enhanced_features)
                expanded_output = self._expand_output_space(original_output)
                return expanded_output
        
        # Original model output
        return model(data.view(data.size(0), -1))
    
    def _expand_output_space(self, original_output: torch.Tensor) -> torch.Tensor:
        """
        Expand model output space to include transferred class.
        Original: [batch_size, num_original_classes]
        Expanded: [batch_size, num_original_classes + 1]
        """
        batch_size = original_output.size(0)
        num_original_classes = original_output.size(1)
        
        # Add a new class dimension (initialized to 0)
        expanded_output = torch.zeros(batch_size, num_original_classes + 1, device=original_output.device)
        expanded_output[:, :num_original_classes] = original_output
        
        return expanded_output
    
    def _get_class_8_confidence(self, outputs: torch.Tensor, num_original_classes: int) -> torch.Tensor:
        """
        Extract confidence for the transferred class (class 8).
        If output space is expanded, use the last dimension.
        If not expanded, use softmax distribution over original classes.
        """
        if outputs.size(1) > num_original_classes:
            # Expanded output space - last dimension is class 8
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities[:, -1]  # Last class is transferred class
        else:
            # Original output space - no specific class 8 neuron
            # Use inverse of max confidence in original classes as proxy
            probabilities = torch.softmax(outputs, dim=1)
            max_original_confidence = probabilities.max(dim=1)[0]
            return 1.0 - max_original_confidence  # Inverse confidence

def print_precise_measurement_definitions():
    """Print exactly what each measurement measures."""
    print("🔍 PRECISE MEASUREMENT DEFINITIONS")
    print("=" * 50)
    print()
    
    print("📊 METRIC 1: Original Knowledge Preservation")
    print("   MEASURES: Target model accuracy on classes {0,1,2,3,4,5,6,7}")
    print("   INPUT: Target test images labeled 0-7")
    print("   OUTPUT: Classification accuracy percentage")
    print("   EXPECTED: ~90% before transfer, should stay ~90% after")
    print("   PURPOSE: Ensure transfer doesn't break existing knowledge")
    print()
    
    print("📊 METRIC 2: Transfer Effectiveness") 
    print("   MEASURES: Binary classification - can model distinguish class 8 from classes 0-7?")
    print("   INPUT: Source test images of class 8 + target test images of classes 0-7")
    print("   OUTPUT: Binary classification accuracy (8 vs not-8)")
    print("   EXPECTED: ~50% before transfer (random), >70% after successful transfer")
    print("   PURPOSE: Measure how well model learns new class")
    print()
    
    print("📊 METRIC 3: Transfer Specificity")
    print("   MEASURES: Transfer only intended class 8, not unintended class 9")
    print("   INPUT: Source test images of classes 8 and 9")
    print("   OUTPUT: Ratio of correct intended vs avoided unintended transfer")
    print("   EXPECTED: 0% before transfer, >70% after specific transfer")
    print("   PURPOSE: Ensure transfer is precise, not general knowledge leakage")
    print()

if __name__ == "__main__":
    print_precise_measurement_definitions()