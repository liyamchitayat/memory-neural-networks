"""
Corrected Metrics Implementation
Following the updated General Requirements with clear metric definitions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Set
from dataclasses import dataclass

@dataclass
class CorrectedTransferMetrics:
    """Corrected metrics following updated requirements."""
    original_knowledge_preservation: float  # Metric 1: Preserve original training data recognition
    transfer_specificity: float             # Metric 2: Transfer only intended class, not others
    transfer_effectiveness: float           # Metric 3: How well transferred class is recognized
    
    def to_dict(self) -> dict:
        return {
            'original_knowledge_preservation': float(self.original_knowledge_preservation),
            'transfer_specificity': float(self.transfer_specificity), 
            'transfer_effectiveness': float(self.transfer_effectiveness)
        }

class CorrectedMetricsEvaluator:
    """Evaluates transfer learning metrics with corrected definitions."""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_transfer_metrics(self, model: nn.Module, transfer_system, 
                                source_test_loader: DataLoader, target_test_loader: DataLoader,
                                transfer_class: int, source_classes: Set[int], target_classes: Set[int]) -> CorrectedTransferMetrics:
        """
        Evaluate all corrected transfer metrics.
        
        Args:
            model: Target model (possibly modified)
            transfer_system: Transfer system (None for "before" measurements)
            source_test_loader: Test data for source classes
            target_test_loader: Test data for target classes
            transfer_class: Specific class being transferred (e.g., 8)
            source_classes: All source model classes (e.g., {2,3,4,5,6,7,8,9})
            target_classes: All target model classes (e.g., {0,1,2,3,4,5,6,7})
            
        Returns:
            CorrectedTransferMetrics with all measured values
        """
        device = self.config.device
        model.eval()
        
        # Metric 1: Original Knowledge Preservation
        original_preservation = self._evaluate_original_knowledge_preservation(
            model, transfer_system, target_test_loader, target_classes, transfer_class)
        
        # Metric 2: Transfer Specificity  
        transfer_specificity = self._evaluate_transfer_specificity(
            model, transfer_system, source_test_loader, transfer_class, source_classes, target_classes)
        
        # Metric 3: Transfer Effectiveness
        transfer_effectiveness = self._evaluate_transfer_effectiveness(
            model, transfer_system, source_test_loader, transfer_class)
        
        return CorrectedTransferMetrics(
            original_knowledge_preservation=original_preservation,
            transfer_specificity=transfer_specificity,
            transfer_effectiveness=transfer_effectiveness
        )
    
    def _evaluate_original_knowledge_preservation(self, model: nn.Module, transfer_system,
                                                target_test_loader: DataLoader, target_classes: Set[int], 
                                                transfer_class: int) -> float:
        """
        Metric 1: Does the target model still recognize its original training data?
        
        REQUIREMENT: Must maintain >80% accuracy on original classes after transfer.
        """
        device = self.config.device
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in target_test_loader:
                data, target = data.to(device), target.to(device)
                
                # Only evaluate on original target classes (the classes it was trained on)
                original_mask = torch.tensor([t.item() in target_classes for t in target])
                if original_mask.sum() == 0:
                    continue
                    
                original_data = data[original_mask]
                original_targets = target[original_mask]
                
                # Test model performance on original data
                if transfer_system is not None:
                    # After transfer: Use enhanced model but test on original classes
                    enhanced_outputs = transfer_system.transfer_concept(original_data, transfer_class)
                    if enhanced_outputs is not None:
                        _, predicted = torch.max(enhanced_outputs, 1)
                    else:
                        # Fallback to original model
                        data_flat = original_data.view(original_data.size(0), -1)
                        outputs = model(data_flat)
                        _, predicted = torch.max(outputs, 1)
                else:
                    # Before transfer: Use original model
                    data_flat = original_data.view(original_data.size(0), -1)
                    outputs = model(data_flat)
                    _, predicted = torch.max(outputs, 1)
                
                # Count correct predictions on original classes
                correct += (predicted == original_targets).sum().item()
                total += original_targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_transfer_specificity(self, model: nn.Module, transfer_system,
                                     source_test_loader: DataLoader, transfer_class: int,
                                     source_classes: Set[int], target_classes: Set[int]) -> float:
        """
        Metric 2: Is transfer specific to intended class, or does it leak other knowledge?
        
        Measures: Intended transfer accuracy vs. unintended transfer accuracy.
        Good specificity means high accuracy on intended class, low on others.
        """
        if transfer_system is None:
            # Before transfer: no specificity to measure (should be 0 for all)
            return 0.0
        
        device = self.config.device
        
        # Get other source classes (potential for unintended transfer)
        other_source_classes = source_classes - {transfer_class} - target_classes
        
        if len(other_source_classes) == 0:
            # No other source classes to leak, perfect specificity
            return 1.0
        
        intended_correct = 0
        intended_total = 0
        unintended_correct = 0  
        unintended_total = 0
        
        with torch.no_grad():
            for data, target in source_test_loader:
                data, target = data.to(device), target.to(device)
                
                # Test intended transfer (target class)
                intended_mask = (target == transfer_class)
                if intended_mask.sum() > 0:
                    intended_data = data[intended_mask]
                    enhanced_outputs = transfer_system.transfer_concept(intended_data, transfer_class)
                    if enhanced_outputs is not None:
                        _, predicted = torch.max(enhanced_outputs, 1)
                        intended_correct += (predicted == transfer_class).sum().item()
                        intended_total += intended_data.size(0)
                
                # Test unintended transfer (other source classes)
                unintended_mask = torch.tensor([t.item() in other_source_classes for t in target])
                if unintended_mask.sum() > 0:
                    unintended_data = data[unintended_mask]
                    unintended_targets = target[unintended_mask]
                    enhanced_outputs = transfer_system.transfer_concept(unintended_data, transfer_class)
                    if enhanced_outputs is not None:
                        _, predicted = torch.max(enhanced_outputs, 1)
                        # Count how many are incorrectly predicted as other source classes
                        for pred, true_target in zip(predicted, unintended_targets):
                            if pred.item() in other_source_classes:
                                unintended_correct += 1
                            unintended_total += 1
        
        # Calculate specificity ratio
        intended_accuracy = intended_correct / intended_total if intended_total > 0 else 0.0
        unintended_accuracy = unintended_correct / unintended_total if unintended_total > 0 else 0.0
        
        # Specificity: intended should be high, unintended should be low
        if intended_accuracy + unintended_accuracy == 0:
            return 0.0
        
        specificity_ratio = intended_accuracy / (intended_accuracy + unintended_accuracy + 1e-8)
        return specificity_ratio
    
    def _evaluate_transfer_effectiveness(self, model: nn.Module, transfer_system,
                                       source_test_loader: DataLoader, transfer_class: int) -> float:
        """
        Metric 3: How well does the target model recognize the transferred class?
        
        Expected: 0% before transfer, high% after transfer.
        """
        device = self.config.device
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in source_test_loader:
                # Only evaluate on transfer class samples
                transfer_mask = (target == transfer_class)
                if transfer_mask.sum() == 0:
                    continue
                    
                transfer_data = data[transfer_mask].to(device)
                transfer_targets = target[transfer_mask].to(device)
                
                if transfer_system is not None:
                    # After transfer: Use enhanced model
                    enhanced_outputs = transfer_system.transfer_concept(transfer_data, transfer_class)
                    if enhanced_outputs is not None:
                        _, predicted = torch.max(enhanced_outputs, 1)
                        correct += (predicted == transfer_class).sum().item()
                        total += transfer_targets.size(0)
                else:
                    # Before transfer: Use original model (should fail)
                    data_flat = transfer_data.view(transfer_data.size(0), -1)
                    outputs = model(data_flat)
                    _, predicted = torch.max(outputs, 1)
                    # This should be ~0% since transfer class wasn't in target training
                    correct += (predicted == transfer_class).sum().item()
                    total += transfer_targets.size(0)
        
        return correct / total if total > 0 else 0.0