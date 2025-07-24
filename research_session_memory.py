#!/usr/bin/env python3
"""
Research Session Memory System
Tracks experiments, results, and research progress for SAE-based neural surgery
"""

import json
import datetime
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class ExperimentResult:
    """Single experiment result"""
    experiment_id: str
    timestamp: str
    method: str
    architecture_source: str
    architecture_target: str
    transfer_accuracy: float
    preservation_accuracy: float
    specificity_accuracy: float
    hyperparameters: Dict[str, Any]
    notes: str
    success_metrics: Dict[str, bool]
    
class ResearchSessionMemory:
    """Persistent memory system for research sessions"""
    
    def __init__(self, memory_file: str = "research_memory.json"):
        self.memory_file = memory_file
        self.current_session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory = self.load_memory()
        
    def load_memory(self) -> Dict:
        """Load existing research memory"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "sessions": {},
                "experiments": [],
                "research_insights": [],
                "current_best": {
                    "same_architecture": None,
                    "cross_architecture": None,
                    "preservation_focused": None
                },
                "failed_approaches": [],
                "hyperparameter_trends": {}
            }
    
    def save_memory(self):
        """Save current memory to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2, default=str)
        print(f"💾 Research memory saved to {self.memory_file}")
    
    def start_session(self, research_focus: str, goals: List[str]):
        """Start a new research session"""
        session_data = {
            "start_time": datetime.datetime.now().isoformat(),
            "research_focus": research_focus,
            "goals": goals,
            "experiments_run": [],
            "insights_gained": [],
            "next_steps": []
        }
        
        self.memory["sessions"][self.current_session_id] = session_data
        print(f"🚀 Started research session: {self.current_session_id}")
        print(f"Focus: {research_focus}")
        print(f"Goals: {', '.join(goals)}")
        
    def log_experiment(self, result: ExperimentResult):
        """Log an experiment result"""
        result_dict = asdict(result)
        result_dict["session_id"] = self.current_session_id
        
        self.memory["experiments"].append(result_dict)
        self.memory["sessions"][self.current_session_id]["experiments_run"].append(result.experiment_id)
        
        # Update best results
        self._update_best_results(result)
        
        print(f"📊 Experiment logged: {result.experiment_id}")
        print(f"   Transfer: {result.transfer_accuracy:.1f}%, Preservation: {result.preservation_accuracy:.1f}%")
        
    def _update_best_results(self, result: ExperimentResult):
        """Update best results tracking"""
        current_best = self.memory["current_best"]
        
        # Calculate composite score
        score = result.transfer_accuracy * 0.6 + result.preservation_accuracy * 0.4
        
        # Determine category
        if result.architecture_source == result.architecture_target:
            category = "same_architecture"
        elif result.architecture_source != result.architecture_target:
            category = "cross_architecture"
        else:
            category = "preservation_focused"
        
        # Update if better
        if (current_best[category] is None or 
            score > current_best[category].get("score", 0)):
            current_best[category] = {
                "experiment_id": result.experiment_id,
                "score": score,
                "transfer": result.transfer_accuracy,
                "preservation": result.preservation_accuracy,
                "method": result.method
            }
    
    def add_insight(self, insight: str, category: str = "general"):
        """Add a research insight"""
        insight_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "category": category,
            "insight": insight
        }
        
        self.memory["research_insights"].append(insight_data)
        self.memory["sessions"][self.current_session_id]["insights_gained"].append(insight)
        
        print(f"💡 Insight added: {insight}")
    
    def log_failed_approach(self, approach: str, reason: str, details: Dict = None):
        """Log a failed approach to avoid repeating"""
        failure_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "approach": approach,
            "reason": reason,
            "details": details or {}
        }
        
        self.memory["failed_approaches"].append(failure_data)
        print(f"❌ Failed approach logged: {approach} - {reason}")
    
    def suggest_next_experiments(self) -> List[str]:
        """Suggest next experiments based on memory"""
        suggestions = []
        
        # Analyze recent trends
        recent_experiments = self.memory["experiments"][-10:]  # Last 10
        
        if len(recent_experiments) < 3:
            suggestions.extend([
                "Run baseline vector space alignment experiment",
                "Test different SAE concept dimensions (16, 24, 32)",
                "Evaluate cross-architecture transfer capabilities"
            ])
        else:
            # Advanced suggestions based on patterns
            transfer_scores = [exp["transfer_accuracy"] for exp in recent_experiments]
            preservation_scores = [exp["preservation_accuracy"] for exp in recent_experiments]
            
            avg_transfer = np.mean(transfer_scores)
            avg_preservation = np.mean(preservation_scores)
            
            if avg_transfer < 30:
                suggestions.append("Focus on improving transfer: try larger concept dimensions or stronger injection")
            if avg_preservation < 95:
                suggestions.append("Focus on preservation: reduce injection strength or improve free space detection")
            if avg_transfer > 40 and avg_preservation > 95:
                suggestions.append("Optimize for extreme architectures: test 64x dimension differences")
            
            # Check for unexplored hyperparameter combinations
            tested_concept_dims = set(exp["hyperparameters"].get("concept_dim", 20) for exp in recent_experiments)
            if 32 not in tested_concept_dims:
                suggestions.append("Test concept_dim=32 for richer concept representations")
            if 12 not in tested_concept_dims:
                suggestions.append("Test concept_dim=12 for more constrained concept space")
        
        return suggestions
    
    def get_hyperparameter_recommendations(self) -> Dict[str, Any]:
        """Get hyperparameter recommendations based on successful experiments"""
        successful_experiments = [
            exp for exp in self.memory["experiments"] 
            if exp["transfer_accuracy"] > 25 and exp["preservation_accuracy"] > 90
        ]
        
        if not successful_experiments:
            return {
                "concept_dim": 20,
                "sparsity_weight": 0.05,
                "injection_strength": 0.3,
                "preservation_weight": 0.9,
                "learning_rate": 0.01
            }
        
        # Analyze successful hyperparameters
        recommendations = {}
        for param in ["concept_dim", "sparsity_weight", "injection_strength"]:
            values = [exp["hyperparameters"].get(param) for exp in successful_experiments if exp["hyperparameters"].get(param) is not None]
            if values:
                recommendations[param] = np.mean(values)
        
        return recommendations
    
    def generate_research_report(self) -> str:
        """Generate a comprehensive research report"""
        report = []
        report.append("=" * 60)
        report.append("RESEARCH SESSION REPORT")
        report.append("=" * 60)
        
        # Session overview
        current_session = self.memory["sessions"].get(self.current_session_id, {})
        report.append(f"\nCurrent Session: {self.current_session_id}")
        report.append(f"Focus: {current_session.get('research_focus', 'Not specified')}")
        report.append(f"Experiments Run: {len(current_session.get('experiments_run', []))}")
        
        # Best results
        report.append(f"\n🏆 BEST RESULTS:")
        for category, result in self.memory["current_best"].items():
            if result:
                report.append(f"  {category.replace('_', ' ').title()}: {result['transfer']:.1f}% transfer, {result['preservation']:.1f}% preservation ({result['method']})")
            else:
                report.append(f"  {category.replace('_', ' ').title()}: No results yet")
        
        # Recent insights
        recent_insights = [ins for ins in self.memory["research_insights"] if ins["session_id"] == self.current_session_id]
        if recent_insights:
            report.append(f"\n💡 SESSION INSIGHTS:")
            for insight in recent_insights[-5:]:  # Last 5
                report.append(f"  • {insight['insight']}")
        
        # Experiment trends
        if len(self.memory["experiments"]) >= 3:
            recent = self.memory["experiments"][-5:]
            avg_transfer = np.mean([exp["transfer_accuracy"] for exp in recent])
            avg_preservation = np.mean([exp["preservation_accuracy"] for exp in recent])
            report.append(f"\n📈 RECENT PERFORMANCE:")
            report.append(f"  Average Transfer: {avg_transfer:.1f}%")
            report.append(f"  Average Preservation: {avg_preservation:.1f}%")
        
        # Next steps
        suggestions = self.suggest_next_experiments()
        if suggestions:
            report.append(f"\n🎯 SUGGESTED NEXT EXPERIMENTS:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                report.append(f"  {i}. {suggestion}")
        
        # Failed approaches to avoid
        recent_failures = [f for f in self.memory["failed_approaches"] if f["session_id"] == self.current_session_id]
        if recent_failures:
            report.append(f"\n⚠️  APPROACHES TO AVOID:")
            for failure in recent_failures[-3:]:
                report.append(f"  • {failure['approach']}: {failure['reason']}")
        
        return "\n".join(report)
    
    def end_session(self, summary: str, next_session_goals: List[str] = None):
        """End current research session"""
        session_data = self.memory["sessions"][self.current_session_id]
        session_data["end_time"] = datetime.datetime.now().isoformat()
        session_data["summary"] = summary
        session_data["next_steps"] = next_session_goals or []
        
        print(f"🏁 Session {self.current_session_id} ended")
        print(f"Summary: {summary}")
        if next_session_goals:
            print(f"Next goals: {', '.join(next_session_goals)}")

def create_experiment_result(experiment_id: str, method: str, arch_source: str, arch_target: str,
                           transfer_acc: float, preservation_acc: float, specificity_acc: float,
                           hyperparams: Dict, notes: str = "") -> ExperimentResult:
    """Helper function to create experiment results"""
    
    # Calculate success metrics
    success_metrics = {
        "transfer_success": transfer_acc > 20,
        "preservation_success": preservation_acc > 90,
        "specificity_success": specificity_acc < 15,
        "overall_success": transfer_acc > 20 and preservation_acc > 90 and specificity_acc < 15
    }
    
    return ExperimentResult(
        experiment_id=experiment_id,
        timestamp=datetime.datetime.now().isoformat(),
        method=method,
        architecture_source=arch_source,
        architecture_target=arch_target,
        transfer_accuracy=transfer_acc,
        preservation_accuracy=preservation_acc,
        specificity_accuracy=specificity_acc,
        hyperparameters=hyperparams,
        notes=notes,
        success_metrics=success_metrics
    )

if __name__ == "__main__":
    # Example usage
    memory = ResearchSessionMemory()
    
    # Start a session
    memory.start_session(
        research_focus="Improving SAE-based knowledge transfer",
        goals=[
            "Achieve >50% transfer accuracy",
            "Maintain >95% preservation",
            "Test cross-architecture capabilities"
        ]
    )
    
    # Log some example experiments
    result1 = create_experiment_result(
        experiment_id="sae_vector_001",
        method="Vector Space Alignment",
        arch_source="WideNN",
        arch_target="WideNN", 
        transfer_acc=49.6,
        preservation_acc=97.0,
        specificity_acc=8.2,
        hyperparams={"concept_dim": 20, "sparsity_weight": 0.05, "injection_strength": 0.3},
        notes="Breakthrough result with free space injection"
    )
    
    memory.log_experiment(result1)
    
    # Add insights
    memory.add_insight("Free space injection prevents interference with existing knowledge", "methodology")
    memory.add_insight("Training depth is more important than architectural similarity", "theoretical")
    
    # Generate report
    print(memory.generate_research_report())
    
    # Save memory
    memory.save_memory()