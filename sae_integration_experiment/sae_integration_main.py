"""
SAE Direct Integration Experiment - Main Runner
Complete separate experiment testing direct SAE integration vs rho blending.

This experiment creates a fully independent test of architectural approaches:
1. Direct SAE integration into model forward pass
2. Comparison with original rho blending approach
3. Separate results directory and files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer

class DirectSAEModel(nn.Module):
    """
    Model with SAE directly integrated into forward pass (no rho blending).
    
    Three integration strategies:
    1. REPLACE: SAE features replace original features entirely
    2. ADD: SAE features added to original features  
    3. CONCAT: SAE features concatenated with original features
    """
    
    def __init__(self, base_model, concept_dim=24):
        super().__init__()
        self.base_model = base_model
        self.concept_dim = concept_dim
        
        # Get feature dimension from base model
        sample_input = torch.randn(1, 784)
        with torch.no_grad():
            features = self.base_model.get_features(sample_input)
            self.feature_dim = features.shape[1]
        
        # Create SAE
        self.sae = SparseAutoencoder(self.feature_dim, concept_dim)
        
        # Integration mode
        self.integration_mode = "replace"
        self.use_injection = False
        self.injection_vector = None
        self.injection_strength = 0.5
        
        # Store original final layer for concat mode
        self.original_final_layer = None
        
    def set_integration_mode(self, mode: str):
        """Set integration mode: 'replace', 'add', or 'concat'."""
        assert mode in ["replace", "add", "concat"]
        self.integration_mode = mode
        
        if mode == "concat" and self.original_final_layer is None:
            # Store original and create new final layer for concatenated features
            self.original_final_layer = self.base_model.fc6
            
            # New final layer with double input size
            input_dim = self.feature_dim * 2
            output_dim = self.original_final_layer.out_features
            
            self.base_model.fc6 = nn.Linear(input_dim, output_dim)
            
            # Initialize with original weights
            with torch.no_grad():
                self.base_model.fc6.weight[:, :self.feature_dim] = self.original_final_layer.weight
                self.base_model.fc6.weight[:, self.feature_dim:] = self.original_final_layer.weight * 0.1
                self.base_model.fc6.bias = self.original_final_layer.bias
                
    def set_concept_injection(self, injection_vector: torch.Tensor, strength: float = 0.5):
        """Set concept injection parameters."""
        self.injection_vector = injection_vector
        self.injection_strength = strength
        self.use_injection = True
        
    def forward(self, x):
        """Forward pass with direct SAE integration."""
        # Get original features
        original_features = self.base_model.get_features(x)
        
        # SAE processing
        concepts = self.sae.encode(original_features)
        
        # Apply concept injection if enabled
        if self.use_injection and self.injection_vector is not None:
            # Broadcast injection vector to batch size
            batch_size = concepts.shape[0]
            injection = self.injection_vector.unsqueeze(0).expand(batch_size, -1)
            concepts = concepts + self.injection_strength * injection
        
        # Decode back to features
        sae_features = self.sae.decode(concepts)
        
        # Integration strategies
        if self.integration_mode == "replace":
            final_features = sae_features
        elif self.integration_mode == "add":
            final_features = original_features + sae_features
        elif self.integration_mode == "concat":
            final_features = torch.cat([original_features, sae_features], dim=1)
        
        # Final classification
        return self.base_model.classify_from_features(final_features)
    
    def get_features(self, x):
        """Get features for compatibility."""
        return self.base_model.get_features(x)
    
    def train_sae(self, data_loader, epochs=30):
        """Train the SAE component."""
        optimizer = optim.Adam(self.sae.parameters(), lr=0.001)
        
        print(f"Training SAE for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= 15:  # Limit for speed
                    break
                
                # Get features from base model
                with torch.no_grad():
                    features = self.base_model.get_features(data)
                
                # Train SAE
                optimizer.zero_grad()
                loss, metrics = self.sae.compute_loss(features)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / count if count > 0 else 0
                print(f"  Epoch {epoch+1}: SAE Loss = {avg_loss:.4f}")
        
        print("✅ SAE training completed")

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for concept extraction."""
    
    def __init__(self, input_dim: int, concept_dim: int, sparsity_weight: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.sparsity_weight = sparsity_weight
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, concept_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, input_dim),
            nn.ReLU()
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(self, x):
        x_recon, z = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Sparsity loss
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }
        
        return total_loss, metrics

def extract_concept_vector(model, data_loader, target_class, concept_dim):
    """Extract concept vector for target class."""
    print(f"Extracting concept vector for class {target_class}...")
    
    model.eval()
    class_concepts = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            mask = (labels == target_class)
            if mask.sum() > 0:
                class_data = data[mask]
                concepts = model.sae.encode(model.get_features(class_data))
                class_concepts.append(concepts.mean(dim=0))
    
    if class_concepts:
        avg_concept = torch.stack(class_concepts).mean(dim=0)
        print(f"✅ Extracted concept vector: shape {avg_concept.shape}")
        return avg_concept
    else:
        print(f"❌ No samples found for class {target_class}")
        return torch.zeros(concept_dim)

def evaluate_model_performance(model, test_loader, target_class, original_classes):
    """Evaluate model performance on different class groups."""
    model.eval()
    
    # Track predictions by class group
    transfer_correct = 0
    transfer_total = 0
    original_correct = 0
    original_total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # Transfer class performance
            transfer_mask = (labels == target_class)
            if transfer_mask.sum() > 0:
                transfer_preds = predicted[transfer_mask]
                transfer_labels = labels[transfer_mask]
                transfer_correct += (transfer_preds == transfer_labels).sum().item()
                transfer_total += transfer_mask.sum().item()
            
            # Original classes performance
            original_mask = torch.tensor([l.item() in original_classes for l in labels])
            if original_mask.sum() > 0:
                original_preds = predicted[original_mask]
                original_labels = labels[original_mask]
                original_correct += (original_preds == original_labels).sum().item()
                original_total += original_mask.sum().item()
    
    transfer_acc = transfer_correct / transfer_total if transfer_total > 0 else 0.0
    original_acc = original_correct / original_total if original_total > 0 else 0.0
    
    return transfer_acc, original_acc

def run_sae_integration_experiment():
    """Run complete SAE integration experiment."""
    
    print("🧪 SAE DIRECT INTEGRATION EXPERIMENT")
    print("=" * 70)
    
    # Create results directory
    results_dir = Path("sae_integration_experiment/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment configuration
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    transfer_class = 8
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=3,  # Fast training for comparison
        batch_size=32,
        learning_rate=0.001,
        concept_dim=24,
        device='cpu'
    )
    
    # Setup data and training
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    print("\n📚 TRAINING BASE MODELS")
    print("-" * 50)
    
    # Train source model
    print("Training source model...")
    source_model = WideNN()
    trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
    
    # Train target model  
    print("Training target model...")
    target_model = WideNN()
    trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
    
    if trained_source is None or trained_target is None:
        print("❌ Model training failed")
        return
    
    print(f"✅ Base models trained successfully")
    print(f"   Source accuracy: {source_acc:.3f}")
    print(f"   Target accuracy: {target_acc:.3f}")
    
    # Test different integration modes
    integration_modes = ["replace", "add", "concat"]
    injection_strengths = [0.3, 0.5, 0.8]
    
    all_results = []
    
    print(f"\n🔧 TESTING SAE INTEGRATION MODES")
    print("-" * 50)
    
    for mode in integration_modes:
        print(f"\n🔹 Testing {mode.upper()} integration mode...")
        
        for strength in injection_strengths:
            print(f"   Injection strength: {strength}")
            
            # Create direct SAE model
            direct_model = DirectSAEModel(trained_target, config.concept_dim)
            direct_model.set_integration_mode(mode)
            
            # Train SAE on target data
            direct_model.train_sae(target_train_loader, epochs=20)
            
            # Extract concept from source model
            # First create a simple SAE for source model to get concept
            source_sae_model = DirectSAEModel(trained_source, config.concept_dim)
            source_sae_model.train_sae(source_train_loader, epochs=20)
            
            # Extract concept vector for transfer class
            concept_vector = extract_concept_vector(
                source_sae_model, source_train_loader, transfer_class, config.concept_dim)
            
            # Set up injection
            direct_model.set_concept_injection(concept_vector, strength)
            
            # Evaluate performance
            # Test on source data for transfer class
            transfer_acc, _ = evaluate_model_performance(
                direct_model, source_test_loader, transfer_class, set())
            
            # Test on target data for original classes
            _, original_acc = evaluate_model_performance(
                direct_model, target_test_loader, transfer_class, target_classes)
            
            # Store results
            result = {
                'integration_mode': mode,
                'injection_strength': strength,
                'transfer_class_accuracy': transfer_acc,
                'original_classes_accuracy': original_acc,
                'meets_preservation_req': original_acc >= 0.8,
                'meets_effectiveness_req': transfer_acc >= 0.7,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            print(f"      Transfer accuracy: {transfer_acc:.3f}")
            print(f"      Original accuracy: {original_acc:.3f}")
            print(f"      Preservation req: {'✅' if original_acc >= 0.8 else '❌'}")
            print(f"      Effectiveness req: {'✅' if transfer_acc >= 0.7 else '❌'}")
    
    # Save detailed results
    results_file = results_dir / "sae_integration_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Generate summary
    generate_integration_summary(all_results, results_dir)
    
    # Find best configuration
    best_result = max(all_results, key=lambda r: r['transfer_class_accuracy'] + r['original_classes_accuracy'])
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Mode: {best_result['integration_mode'].upper()}")
    print(f"   Strength: {best_result['injection_strength']}")
    print(f"   Transfer accuracy: {best_result['transfer_class_accuracy']:.3f}")
    print(f"   Original accuracy: {best_result['original_classes_accuracy']:.3f}")
    
    return all_results

def generate_integration_summary(results, results_dir):
    """Generate comprehensive summary of integration experiment."""
    
    print(f"\n📊 GENERATING INTEGRATION SUMMARY")
    print("-" * 50)
    
    # Group results by integration mode
    by_mode = {}
    for result in results:
        mode = result['integration_mode']
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(result)
    
    # Create summary
    summary = {
        'experiment_name': 'SAE Direct Integration vs Rho Blending',
        'description': 'Comparison of direct SAE integration modes against traditional rho blending',
        'total_configurations': len(results),
        'timestamp': datetime.now().isoformat(),
        'modes_tested': list(by_mode.keys()),
        'injection_strengths_tested': sorted(list(set(r['injection_strength'] for r in results))),
        'mode_analysis': {}
    }
    
    # Analyze each mode
    for mode, mode_results in by_mode.items():
        best_result = max(mode_results, key=lambda r: r['transfer_class_accuracy'])
        avg_transfer = np.mean([r['transfer_class_accuracy'] for r in mode_results])
        avg_original = np.mean([r['original_classes_accuracy'] for r in mode_results])
        
        mode_analysis = {
            'best_configuration': {
                'injection_strength': best_result['injection_strength'],
                'transfer_accuracy': best_result['transfer_class_accuracy'],
                'original_accuracy': best_result['original_classes_accuracy']
            },
            'average_performance': {
                'transfer_accuracy': avg_transfer,
                'original_accuracy': avg_original
            },
            'configurations_meeting_requirements': len([
                r for r in mode_results 
                if r['meets_preservation_req'] and r['meets_effectiveness_req']
            ])
        }
        
        summary['mode_analysis'][mode] = mode_analysis
    
    # Save summary
    summary_file = results_dir / "sae_integration_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate comparison report
    generate_comparison_report(summary, results_dir)
    
    print(f"✅ Summary saved to: {summary_file}")

def generate_comparison_report(summary, results_dir):
    """Generate human-readable comparison report."""
    
    report_file = results_dir / "SAE_INTEGRATION_vs_RHO_BLENDING_REPORT.md"
    
    with open(report_file, 'w') as f:
        f.write("# SAE Direct Integration vs Rho Blending - Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Overview\n\n")
        f.write("This experiment tests an alternative architectural approach to neural concept transfer:\n\n")
        f.write("**Traditional Approach (Rho Blending):**\n")
        f.write("```\n")
        f.write("final_features = ρ * original_features + (1-ρ) * enhanced_features\n")
        f.write("```\n\n")
        
        f.write("**Direct Integration Approach:**\n")
        f.write("- REPLACE: Use SAE features directly (no original features)\n")
        f.write("- ADD: Add SAE features to original features\n")
        f.write("- CONCAT: Concatenate SAE and original features\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Integration Mode | Best Transfer Acc | Best Original Acc | Configs Meeting Reqs |\n")
        f.write("|------------------|-------------------|--------------------|-----------------------|\n")
        
        for mode, analysis in summary['mode_analysis'].items():
            best = analysis['best_configuration']
            meeting_reqs = analysis['configurations_meeting_requirements']
            f.write(f"| {mode.upper():<16} | {best['transfer_accuracy']:<17.3f} | {best['original_accuracy']:<18.3f} | {meeting_reqs:<21} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find best overall mode
        best_mode = max(summary['mode_analysis'].keys(), 
                       key=lambda m: summary['mode_analysis'][m]['best_configuration']['transfer_accuracy'])
        
        f.write(f"### Best Integration Mode: {best_mode.upper()}\n\n")
        best_config = summary['mode_analysis'][best_mode]['best_configuration']
        f.write(f"- **Transfer Accuracy:** {best_config['transfer_accuracy']:.1%}\n")
        f.write(f"- **Original Accuracy:** {best_config['original_accuracy']:.1%}\n")
        f.write(f"- **Injection Strength:** {best_config['injection_strength']}\n\n")
        
        f.write("### Comparison with Rho Blending\n\n")
        f.write("**Theoretical Rho Blending Results (from main experiment):**\n")
        f.write("- Transfer Accuracy: 72.5%\n")
        f.write("- Original Accuracy: 83.4%\n")
        f.write("- Meets both requirements: ✅\n\n")
        
        f.write("**Direct Integration Results:**\n")
        f.write(f"- Best Transfer Accuracy: {best_config['transfer_accuracy']:.1%}\n")
        f.write(f"- Best Original Accuracy: {best_config['original_accuracy']:.1%}\n\n")
        
        # Determine winner
        rho_score = 0.725 + 0.834  # Rho blending scores
        direct_score = best_config['transfer_accuracy'] + best_config['original_accuracy']
        
        if direct_score > rho_score:
            f.write("**Winner: Direct Integration** 🎉\n")
            f.write(f"Direct integration achieves better overall performance ({direct_score:.3f} vs {rho_score:.3f})\n\n")
        else:
            f.write("**Winner: Rho Blending** 🏆\n")
            f.write(f"Rho blending maintains superior balanced performance ({rho_score:.3f} vs {direct_score:.3f})\n\n")
        
        f.write("## Technical Analysis\n\n")
        f.write("### REPLACE Mode\n")
        f.write("- Completely replaces original features with SAE reconstructions\n")
        f.write("- Risk: May lose important original information\n")
        f.write("- Benefit: Clean separation between original and injected concepts\n\n")
        
        f.write("### ADD Mode\n")
        f.write("- Adds SAE features to original features\n")
        f.write("- Risk: May cause feature magnitude issues\n")
        f.write("- Benefit: Preserves all original information\n\n")
        
        f.write("### CONCAT Mode\n")
        f.write("- Concatenates original and SAE features\n")
        f.write("- Risk: Doubles feature dimensionality\n")
        f.write("- Benefit: Allows model to learn optimal combination\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("This experiment provides valuable insights into architectural choices for neural concept transfer:\n\n")
        f.write("1. **Architectural Flexibility:** Direct integration offers more architectural control\n")
        f.write("2. **Performance Tradeoffs:** Different integration modes have distinct characteristics\n")
        f.write("3. **Complexity vs Performance:** Rho blending may offer better balance of simplicity and performance\n\n")
        
        f.write("## Future Work\n\n")
        f.write("- Test with larger models and datasets\n")
        f.write("- Explore learned integration weights\n")
        f.write("- Investigate computational efficiency differences\n")
        f.write("- Study gradient flow in different integration modes\n")
    
    print(f"📝 Comparison report generated: {report_file}")

if __name__ == "__main__":
    print("🚀 Starting SAE Direct Integration Experiment...")
    print("This experiment tests architectural alternatives to rho blending")
    print("Results will be saved in sae_integration_experiment/results/")
    print("")
    print("⚠️  Make sure you're in the 'neural_transfer' conda environment:")
    print("   conda activate neural_transfer")
    print("")
    
    results = run_sae_integration_experiment()
    
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print(f"📁 Results available in: sae_integration_experiment/results/")
    print(f"📊 Files generated:")
    print(f"   • sae_integration_detailed_results.json - Raw results")
    print(f"   • sae_integration_summary.json - Analysis summary")
    print(f"   • SAE_INTEGRATION_vs_RHO_BLENDING_REPORT.md - Comparison report")
    print(f"\n🔬 This provides valuable architectural insights for neural concept transfer!")