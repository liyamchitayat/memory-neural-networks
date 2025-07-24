#!/usr/bin/env python3
"""
SAE Research Planner
Systematic research plan for improving SAE-based neural surgery
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from research_session_memory import ResearchSessionMemory, create_experiment_result
import itertools

@dataclass
class ResearchHypothesis:
    """Research hypothesis to test"""
    hypothesis_id: str
    description: str
    motivation: str
    expected_outcome: str
    experiments_needed: List[str]
    priority: str  # "high", "medium", "low"
    dependencies: List[str]  # Other hypothesis IDs this depends on
    
@dataclass
class ExperimentPlan:
    """Planned experiment"""
    experiment_id: str
    hypothesis_id: str
    method: str
    architecture_pair: Tuple[str, str]
    hyperparameters: Dict[str, Any]
    success_criteria: Dict[str, float]
    estimated_runtime: str
    implementation_notes: str

class SAEResearchPlanner:
    """Systematic research planner for SAE improvements"""
    
    def __init__(self, memory_system: ResearchSessionMemory):
        self.memory = memory_system
        self.hypotheses = self._initialize_research_hypotheses()
        self.experiment_plans = self._generate_experiment_plans()
        
    def _initialize_research_hypotheses(self) -> List[ResearchHypothesis]:
        """Initialize core research hypotheses"""
        
        hypotheses = [
            ResearchHypothesis(
                hypothesis_id="H1_concept_dimension_scaling",
                description="Larger concept dimensions enable better transfer by capturing richer representations",
                motivation="Current 20D may be too restrictive for complex digit patterns",
                expected_outcome="Transfer accuracy increases with concept dimension up to a saturation point",
                experiments_needed=["concept_dim_sweep", "dimension_ablation", "representation_analysis"],
                priority="high",
                dependencies=[]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H2_sparsity_transfer_tradeoff",
                description="Sparsity regularization creates cleaner concept spaces but may limit transfer capacity",
                motivation="Current λ=0.05 may be suboptimal for transfer tasks",
                expected_outcome="Optimal sparsity weight balances interpretability and transfer capacity",
                experiments_needed=["sparsity_sweep", "concept_quality_analysis", "transfer_capacity_test"],
                priority="high",
                dependencies=[]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H3_hierarchical_concepts",
                description="Multi-level concept hierarchies improve transfer over flat concept spaces",
                motivation="Digits have hierarchical structure (strokes → digits → classes)",
                expected_outcome="Hierarchical SAEs achieve better transfer and interpretability",
                experiments_needed=["hierarchical_sae_design", "multi_level_transfer", "concept_hierarchy_analysis"],
                priority="medium",
                dependencies=["H1_concept_dimension_scaling"]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H4_dynamic_concept_alignment",
                description="Learned alignment networks outperform static Procrustes alignment",
                motivation="Static alignment may not capture complex non-linear relationships",
                expected_outcome="Neural alignment networks improve cross-architecture transfer",
                experiments_needed=["alignment_network_comparison", "nonlinear_alignment_test", "alignment_quality_analysis"],
                priority="high",
                dependencies=[]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H5_multi_concept_transfer",
                description="Simultaneous multi-concept transfer is more efficient than sequential transfer",
                motivation="Current approach transfers one concept at a time",
                expected_outcome="Joint transfer of multiple concepts improves overall performance",
                experiments_needed=["multi_digit_transfer", "concept_interference_analysis", "joint_optimization"],
                priority="medium",
                dependencies=["H1_concept_dimension_scaling", "H2_sparsity_transfer_tradeoff"]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H6_adversarial_concept_robustness",
                description="Adversarial training improves concept transfer robustness",
                motivation="Current concepts may be brittle to input variations",
                expected_outcome="Adversarially trained concepts transfer more reliably",
                experiments_needed=["adversarial_concept_training", "robustness_evaluation", "transfer_stability_test"],
                priority="low",
                dependencies=["H1_concept_dimension_scaling"]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H7_architecture_agnostic_concepts", 
                description="Universal concept spaces work across all architectures without alignment",
                motivation="Current approach requires pairwise alignment",
                expected_outcome="Universal SAEs eliminate need for alignment step",
                experiments_needed=["universal_sae_training", "multi_architecture_evaluation", "concept_universality_test"],
                priority="medium",
                dependencies=["H4_dynamic_concept_alignment"]
            ),
            
            ResearchHypothesis(
                hypothesis_id="H8_continual_concept_learning",
                description="SAEs can continuously learn new concepts without forgetting old ones",
                motivation="Current approach requires retraining for new concepts",
                expected_outcome="Continual learning SAEs enable incremental concept addition",
                experiments_needed=["continual_sae_design", "catastrophic_forgetting_test", "incremental_transfer"],
                priority="low",
                dependencies=["H3_hierarchical_concepts", "H5_multi_concept_transfer"]
            )
        ]
        
        return hypotheses
    
    def _generate_experiment_plans(self) -> List[ExperimentPlan]:
        """Generate detailed experiment plans for each hypothesis"""
        
        plans = []
        
        # H1: Concept Dimension Scaling
        for concept_dim in [12, 16, 24, 32, 48, 64]:
            plans.append(ExperimentPlan(
                experiment_id=f"concept_dim_{concept_dim}",
                hypothesis_id="H1_concept_dimension_scaling",
                method="Vector Space Alignment",
                architecture_pair=("WideNN", "WideNN"),
                hyperparameters={
                    "concept_dim": concept_dim,
                    "sparsity_weight": 0.05,
                    "injection_strength": 0.3,
                    "learning_rate": 0.01
                },
                success_criteria={"transfer_accuracy": 30.0, "preservation_accuracy": 90.0},
                estimated_runtime="15 minutes",
                implementation_notes="Test if larger concept dimensions improve transfer capacity"
            ))
        
        # H2: Sparsity-Transfer Tradeoff
        for sparsity in [0.01, 0.03, 0.05, 0.08, 0.12, 0.20]:
            plans.append(ExperimentPlan(
                experiment_id=f"sparsity_{sparsity:.3f}",
                hypothesis_id="H2_sparsity_transfer_tradeoff",
                method="Vector Space Alignment",
                architecture_pair=("WideNN", "WideNN"),
                hyperparameters={
                    "concept_dim": 20,
                    "sparsity_weight": sparsity,
                    "injection_strength": 0.3,
                    "learning_rate": 0.01
                },
                success_criteria={"transfer_accuracy": 25.0, "preservation_accuracy": 92.0},
                estimated_runtime="15 minutes",
                implementation_notes="Find optimal sparsity level for transfer performance"
            ))
        
        # H4: Dynamic Concept Alignment
        alignment_methods = ["procrustes", "linear", "nonlinear_shallow", "nonlinear_deep"]
        for method in alignment_methods:
            plans.append(ExperimentPlan(
                experiment_id=f"alignment_{method}",
                hypothesis_id="H4_dynamic_concept_alignment",
                method="Dynamic Alignment Transfer",
                architecture_pair=("WideNN", "DeepNN"),
                hyperparameters={
                    "concept_dim": 20,
                    "sparsity_weight": 0.05,
                    "alignment_method": method,
                    "alignment_lr": 0.01
                },
                success_criteria={"transfer_accuracy": 20.0, "preservation_accuracy": 90.0},
                estimated_runtime="25 minutes",
                implementation_notes=f"Test {method} alignment for cross-architecture transfer"
            ))
        
        # H3: Hierarchical Concepts
        hierarchy_configs = [
            {"levels": 2, "dims": [32, 16]},
            {"levels": 3, "dims": [48, 24, 12]},
            {"levels": 2, "dims": [40, 20]}
        ]
        for config in hierarchy_configs:
            plans.append(ExperimentPlan(
                experiment_id=f"hierarchical_L{config['levels']}_{'_'.join(map(str, config['dims']))}",
                hypothesis_id="H3_hierarchical_concepts",
                method="Hierarchical SAE Transfer",
                architecture_pair=("WideNN", "WideNN"),
                hyperparameters={
                    "hierarchy_levels": config["levels"],
                    "concept_dims": config["dims"],
                    "sparsity_weight": 0.05,
                    "hierarchy_loss_weight": 0.1
                },
                success_criteria={"transfer_accuracy": 35.0, "preservation_accuracy": 90.0},
                estimated_runtime="30 minutes", 
                implementation_notes="Test hierarchical concept representations"
            ))
        
        # H5: Multi-Concept Transfer
        multi_transfer_configs = [
            {"source_digits": [4], "target_digits": [4]},
            {"source_digits": [4, 5], "target_digits": [4, 5]},
            {"source_digits": [4, 5], "target_digits": [4]},  # Transfer from multi to single
        ]
        for config in multi_transfer_configs:
            plans.append(ExperimentPlan(
                experiment_id=f"multi_transfer_{''.join(map(str, config['source_digits']))}_to_{''.join(map(str, config['target_digits']))}",
                hypothesis_id="H5_multi_concept_transfer",
                method="Multi-Concept Vector Transfer",
                architecture_pair=("WideNN", "WideNN"),
                hyperparameters={
                    "concept_dim": 24,
                    "source_digits": config["source_digits"],
                    "target_digits": config["target_digits"],
                    "multi_concept_weight": 0.8
                },
                success_criteria={"transfer_accuracy": 25.0, "preservation_accuracy": 88.0},
                estimated_runtime="20 minutes",
                implementation_notes="Test simultaneous multi-concept transfer"
            ))
        
        return plans
    
    def get_next_experiments(self, num_experiments: int = 3) -> List[ExperimentPlan]:
        """Get next experiments to run based on priorities and dependencies"""
        
        # Get completed experiments from memory
        completed_experiment_ids = set(exp["experiment_id"] for exp in self.memory.memory["experiments"])
        
        # Filter out completed experiments
        available_plans = [plan for plan in self.experiment_plans 
                          if plan.experiment_id not in completed_experiment_ids]
        
        # Check dependencies
        validated_plans = []
        for plan in available_plans:
            hypothesis = next(h for h in self.hypotheses if h.hypothesis_id == plan.hypothesis_id)
            
            # Check if dependencies are satisfied
            dependencies_satisfied = True
            for dep_id in hypothesis.dependencies:
                dep_experiments = [p.experiment_id for p in self.experiment_plans 
                                 if p.hypothesis_id == dep_id]
                if not any(exp_id in completed_experiment_ids for exp_id in dep_experiments):
                    dependencies_satisfied = False
                    break
            
            if dependencies_satisfied:
                validated_plans.append(plan)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        validated_plans.sort(key=lambda p: (
            priority_order[next(h for h in self.hypotheses if h.hypothesis_id == p.hypothesis_id).priority],
            p.experiment_id
        ))
        
        return validated_plans[:num_experiments]
    
    def generate_research_roadmap(self) -> str:
        """Generate a comprehensive research roadmap"""
        roadmap = []
        roadmap.append("=" * 70)
        roadmap.append("SAE RESEARCH ROADMAP")
        roadmap.append("=" * 70)
        
        # Hypothesis overview
        roadmap.append(f"\n🧠 RESEARCH HYPOTHESES ({len(self.hypotheses)} total):")
        for hypothesis in self.hypotheses:
            status = "✅" if self._hypothesis_completed(hypothesis) else "🔄" if self._hypothesis_in_progress(hypothesis) else "⏳"
            roadmap.append(f"\n{status} {hypothesis.hypothesis_id.upper()}")
            roadmap.append(f"   {hypothesis.description}")
            roadmap.append(f"   Priority: {hypothesis.priority.upper()}")
            if hypothesis.dependencies:
                roadmap.append(f"   Dependencies: {', '.join(hypothesis.dependencies)}")
        
        # Next experiments
        next_experiments = self.get_next_experiments(5)
        roadmap.append(f"\n🎯 NEXT EXPERIMENTS TO RUN:")
        for i, plan in enumerate(next_experiments, 1):
            roadmap.append(f"\n{i}. {plan.experiment_id}")
            roadmap.append(f"   Method: {plan.method}")
            roadmap.append(f"   Architecture: {plan.architecture_pair[0]} → {plan.architecture_pair[1]}")
            roadmap.append(f"   Key params: {dict(list(plan.hyperparameters.items())[:3])}")
            roadmap.append(f"   Runtime: {plan.estimated_runtime}")
        
        # Progress summary
        completed = len([exp for exp in self.memory.memory["experiments"]])
        total_planned = len(self.experiment_plans)
        roadmap.append(f"\n📊 PROGRESS SUMMARY:")
        roadmap.append(f"   Experiments completed: {completed}")
        roadmap.append(f"   Experiments planned: {total_planned}")
        roadmap.append(f"   Progress: {completed/total_planned*100:.1f}%")
        
        # Research phases
        roadmap.append(f"\n📈 RESEARCH PHASES:")
        roadmap.append(f"   Phase 1 (Foundation): Optimize basic SAE parameters")
        roadmap.append(f"   Phase 2 (Architecture): Improve cross-architecture transfer")  
        roadmap.append(f"   Phase 3 (Advanced): Hierarchical and multi-concept methods")
        roadmap.append(f"   Phase 4 (Robustness): Adversarial and continual learning")
        
        return "\n".join(roadmap)
    
    def _hypothesis_completed(self, hypothesis: ResearchHypothesis) -> bool:
        """Check if hypothesis has been adequately tested"""
        completed_experiments = set(exp["experiment_id"] for exp in self.memory.memory["experiments"])
        hypothesis_experiments = [p.experiment_id for p in self.experiment_plans 
                                if p.hypothesis_id == hypothesis.hypothesis_id]
        
        # Consider hypothesis completed if at least 50% of experiments are done
        completion_ratio = len([exp for exp in hypothesis_experiments 
                              if exp in completed_experiments]) / max(len(hypothesis_experiments), 1)
        return completion_ratio >= 0.5
    
    def _hypothesis_in_progress(self, hypothesis: ResearchHypothesis) -> bool:
        """Check if hypothesis testing is in progress"""
        completed_experiments = set(exp["experiment_id"] for exp in self.memory.memory["experiments"])
        hypothesis_experiments = [p.experiment_id for p in self.experiment_plans 
                                if p.hypothesis_id == hypothesis.hypothesis_id]
        
        # Consider in progress if some but not all experiments are done
        completed_count = len([exp for exp in hypothesis_experiments 
                             if exp in completed_experiments])
        return 0 < completed_count < len(hypothesis_experiments)
    
    def get_experiment_plan(self, experiment_id: str) -> Optional[ExperimentPlan]:
        """Get specific experiment plan by ID"""
        for plan in self.experiment_plans:
            if plan.experiment_id == experiment_id:
                return plan
        return None
    
    def update_hypothesis_priority(self, hypothesis_id: str, new_priority: str):
        """Update hypothesis priority based on results"""
        for hypothesis in self.hypotheses:
            if hypothesis.hypothesis_id == hypothesis_id:
                hypothesis.priority = new_priority
                print(f"Updated {hypothesis_id} priority to {new_priority}")
                break

if __name__ == "__main__":
    # Initialize system
    memory = ResearchSessionMemory()
    planner = SAEResearchPlanner(memory)
    
    # Start research session
    memory.start_session(
        research_focus="Systematic SAE improvement research",
        goals=[
            "Test all major hypotheses systematically",
            "Achieve >60% transfer with >95% preservation",
            "Develop robust cross-architecture methods"
        ]
    )
    
    # Generate and display roadmap
    print(planner.generate_research_roadmap())
    
    # Get next experiments
    next_experiments = planner.get_next_experiments(3)
    print(f"\n🚀 RECOMMENDED NEXT EXPERIMENTS:")
    for i, plan in enumerate(next_experiments, 1):
        print(f"\n{i}. {plan.experiment_id}")
        print(f"   Hypothesis: {plan.hypothesis_id}")
        print(f"   Implementation: {plan.implementation_notes}")
        print(f"   Success criteria: {plan.success_criteria}")