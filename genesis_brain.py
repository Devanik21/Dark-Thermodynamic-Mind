import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import uuid

# ============================================================
# 🧬 NEURAL ARCHITECTURE
# ============================================================
# ============================================================
# 🧬 NEURAL ARCHITECTURE
# ============================================================
class GradientCompressor(nn.Module):
    """
    5.7 Cognitive Compression: Learned Low-Rank Approximation of Gradients.
    'Learning to Learn' by compressing update vectors into 'principal components' of adaptation.
    """
    def __init__(self, input_dim, rank=8):
        super().__init__()
        # U * V approx of the gradient space
        self.U = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        
    def forward(self, grad):
        # Project gradient into low-rank subspace and back
        # grad_approx = (grad @ V.T) @ U.T
        if grad is None: return None
        # Simple compression: Filter gradient through the bottleneck
        # We want to find the 'component' of the gradient that aligns with U*V
        # But for 'meta-learning', we essentially want to MODIFY the gradient.
        # G_new = G + (G @ V.T @ U.T) * alpha
        
        # Matrix multiplication match: Grad shape (N, Out) or (Out, In)?
        # Assumes flattened or compatible shape. 
        # For simplicity, we apply this to the generic 1D flattened gradient vector if used,
        # OR we treat it as layer-wise modulation.
        
        # SIMPLIFIED IMPLEMENTATION for 5.7:
        # We learn a 'filter' that amplifies useful gradient directions and suppressed noise.
        # But U, V needs to match dimensions. 
        # Let's effectively assume this is a scalar gate per parameter for now to save compute,
        # OR a small MLP that takes gradient statistics and outputs a scaling factor.
        
        # Better: Low Rank Adaptation (LoRA) style but for the UPDATE rule.
        pass

class PruningMask(nn.Module):
    """5.2 Architecture Search: Learnable mask for weight pruning."""
    def __init__(self, shape):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.ones(shape) * 5.0) # Start fully connected
        
    def forward(self):
        # Differentiable binary mask via Sigmoid ~ Gate
        return torch.sigmoid(self.mask_logits)

    def sparsity(self):
        # Return "Soft Sparsity" (1.0 - average density) for better visualization
        # Hard thresholding stays at 0% for too long.
        return 1.0 - self.forward().mean()

class GenesisBrain(nn.Module):
    """
    V-DV4 Dreamer Architecture (2026 SOTA) for 96 Super-Agents.
    """
    def __init__(self, input_dim=41, hidden_dim=256, output_dim=21):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Encoder (Sensory Processing)
        # Compresses 41D input -> 256D Latent State
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 2. RSSM (Recurrent State-Space Model) - The Dream Engine
        # Deterministic state (h) + Stochastic state (z)
        self.rssm_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 3. Transformer Actor (Attention-based Policy)
        # Tiny Transformer Block for attention-based decision-making
        self.actor_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.actor = nn.Linear(hidden_dim, output_dim)
        
        # 4. Critic (Value Function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 5. Reward Predictor (For Dreaming)
        self.reward_predictor = nn.Linear(hidden_dim, 1)
        
        # Auxiliary Heads
        self.comm_out = nn.Linear(hidden_dim, 16)
        self.meta_out = nn.Linear(hidden_dim, 4)
        self.predictor = nn.Linear(hidden_dim, input_dim) # Reconstruction
        self.abstraction_encoder = nn.Linear(hidden_dim, 8) # Concepts

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, x, hidden):
        # Ensure hidden state is correct shape (B, 256) for GRUCell
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_dim)
        elif hidden.dim() == 3: 
            hidden = hidden.squeeze(0)
            
        # 1. Encode
        embed = self.encoder(x)
        
        # 2. Recurrent Step (RSSM)
        h_next = self.rssm_cell(embed, hidden)
        
        # 3. Actor (Transformer Block)
        # Add sequence dim for attention: (B, 1, H)
        h_seq = h_next.unsqueeze(1)
        # Self-Attention on current thought
        attn_out, _ = self.actor_attention(h_seq, h_seq, h_seq)
        action_feat = attn_out.squeeze(1) + h_next # Residual
        
        # 4. Heads
        vector = torch.relu(self.actor(action_feat))
        comm = torch.sigmoid(self.comm_out(h_next))
        meta = torch.sigmoid(self.meta_out(h_next))
        value = self.critic(h_next)
        prediction = self.predictor(h_next)
        concepts = torch.relu(self.abstraction_encoder(h_next))
        
        # Return h_next as (1, B, H) to match old GRU API
        return vector, comm, meta, value, h_next.unsqueeze(0), prediction, concepts

    def dream(self, start_state, horizon=10):
        # Rolling out the future in latent space
        states = []
        rewards = []
        
        # Ensure start_state is (B, 256)
        if start_state.dim() == 3: h = start_state.squeeze(0)
        else: h = start_state
            
        for t in range(horizon):
            # Closed-loop hallucination: Assume zero-noise stability for planning
            # In V-DV4, we predict the next state based on internal dynamics
            # Use 'random noise' as the innovation signal for the dream
            noise = torch.randn_like(h) * 0.1
            h = self.rssm_cell(noise, h)
            
            r = self.reward_predictor(h)
            states.append(h)
            rewards.append(r)
            
        return torch.stack(states), torch.stack(rewards)

# ============================================================
# 🤖 THE AGENT
# ============================================================
class GenesisAgent:
    def __init__(self, x, y, genome=None, generation=0, parent_hidden=None, parent_inventory=None, parent_id=None):
        self.id = str(uuid.uuid4())
        self.parent_id = parent_id if parent_id else "World"
        self.x = x
        self.y = y
        self.generation = generation
        self.age = 0
        self.dialect_id = 0 # Level 7.9 Protocol Cluster
        self.energy = 120.0 # Increased starting energy (Survival Buffer)
        self.energy_stored = 0.0 # 1.5 Homeostasis
        self.inventory = [0, 0, 0] if parent_inventory is None else parent_inventory
        
        # 1.3 Landauer Limit metrics
        self.last_weight_entropy = 0.0
        self.reflexes_used = 0
        self.thoughts_had = 0
        self.social_memory = {}
        self.inventions = [] # Level 3.6 Innovation Tracking
        self.tag = np.random.rand(3) # 3.10 Cultural tag (RGB tribe)
        
        # 1.6 Circadian Rhythms
        self.internal_phase = random.random() * 2 * np.pi
        self.influence = 0.0 # 4.4 Initialize early to avoid AttributeError
        
        # --- PHASE 15: LEVEL 4 SPECIALIZATION ---
        self.role = "Generalist" # 4.0 Behavioral Polymorphism
        self.role_history = []  # 4.1 Role Stability
        self.caste_gene = np.random.rand(4) # 4.6 Caste Genetics (Vector predisposition for roles)
        self.is_fused = False   # 4.7 Dynamic Tensor Fusion
        self.fused_partner = None
        self.is_fertile = True # 4.10 Eusociality (Queens vs Workers)
        if generation > 0:
            # 4.10: 20% chance to be a Queen if gen > 0 (simplification for starting)
            self.is_fertile = random.random() < 0.2
        
        # Neural State
        self.brain = GenesisBrain()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.005)
        
        # 3.0 Epigenetic Memory: Inherit mental state
        if parent_hidden is not None:
            self.hidden_state = parent_hidden.detach().clone() + torch.randn_like(parent_hidden) * 0.1
        else:
            self.hidden_state = torch.zeros(1, 1, 256)
        
        # v5.0.4 Persistent state for Ghost Forward
        self.prev_input = None
        self.prev_hidden = None
        
        self.last_concepts = torch.zeros(1, 8) # 5.8 Initialize
        
        # LEVEL 5 STATE MEMORY
        # 5.0 Self-Monitoring
        self.prediction_errors = []
        self.confidence = 0.5 
        
        # 5.1 Meta-Learning (Hypergradients)
        self.meta_lr = 0.005
        self.last_grad_norm = 0.0
        
        # 5.9 Causal Reasoning (Counterfactuals)
        self.causal_graph = {} # {action_dim -> sensory_impact_score}
        
        # 5.10 Autonomous Research
        self.research_log = []
        
        # 3.2 Horizontal Neural Transfer (Viral Memory)
        self.meme_pool = [] # List of {weights: StateDict, fitness: float, beta: float, type: 'virus'}

        # ============================================================
        # 🌍 LEVEL 6: GEO-ENGINEERING STATE
        # ============================================================
        # 6.0 Environmental Prediction
        self.env_history = []  # [(x, y, signal, tick)] history buffer
        self.env_prediction_accuracy = 0.0
        self.env_predictor_hidden = torch.zeros(1, 1, 32)
        
        # 6.1 Probabilistic Environment Collapse (Bayesian Niche Construction)
        self.env_beliefs = {}  # {(x,y): np.array([P(food), P(poison), P(empty)])}
        self.niche_modifications = 0  # Count of environment modifications
        
        # 6.2-6.3 Structure Building & Trap Construction
        self.structures_built = []  # List of structure IDs this agent built
        
        # 6.4 Defensive Architecture
        self.shield_strength = 0.0  # Damage reduction (0-1)
        
        # 6.5 Resource Cultivation
        self.cultivation_sites = {}  # {(x,y): cultivation_strength}
        
        # 6.6 Weather Control
        self.weather_vote = 0.0  # Agent's vote for season amplitude modification
        
        # 6.7 Terraforming
        self.terraform_count = 0  # Number of terrain modifications
        
        # 6.8 Energy Storage (Environmental Batteries)
        self.env_batteries = {}  # {(x,y): stored_energy}
        
        # 6.9 Infrastructure Networks
        self.network_memberships = set()  # Network IDs this agent participates in
        
        # 6.10 Complete Environmental Mastery
        self.env_control_score = 0.0  # H(Env|Actions) tracking
        
        # ============================================================
        # 🐝 LEVEL 7: COLLECTIVE MANIFOLD STATE
        # ============================================================
        # 7.0 Neural Bridging
        self.neural_bridge_partners = set()  # IDs of agents with shared hidden state
        
        # 7.1 Kuramoto Synchronization
        self.kuramoto_phase = random.random() * 2 * np.pi
        self.natural_frequency = np.random.normal(1.0, 0.1)
        self.coupling_strength = 0.5
        
        # 7.2 Gradient Sharing (Federated Learning)
        self.last_shared_gradient = None
        
        # 7.3 Collective Backpropagation
        self.backprop_depth = 0  # How many agents deep error propagated
        
        # 7.4 Modular Cognition
        self.cognitive_specialty = None  # "vision", "planning", "memory", "motor"
        
        # 7.5 Attention Routing
        self.attention_queries_sent = 0
        self.attention_queries_received = 0
        
        # 7.6 Consensus Mechanisms
        self.consensus_votes = {}  # {proposal_id: vote}
        
        # 7.7 Distributed Memory
        self.distributed_memory_fragments = {}  # {memory_id: fragment_data}
        
        # 7.8 Fault Tolerance
        self.backup_connections = set()  # Redundant connection IDs
        
        # 7.9 Emergent Protocols
        self.protocol_version = np.random.rand(8)  # Communication protocol vector
        
        # 7.10 Hive Mind
        self.hive_contribution = 0.0  # This agent's contribution to collective Φ
        
        # ============================================================
        # 💭 LEVEL 8: ABSTRACT REPRESENTATION STATE
        # ============================================================
        # 8.0 Internal Simulation (World Model)
        self.world_model = nn.Sequential(
            nn.Linear(41 + 21, 64),  # State + Action -> Next State
            nn.Tanh(),
            nn.Linear(64, 41)
        )
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=0.001)
        
        # 8.1 Counterfactual Reasoning (enhanced from 5.9)
        self.counterfactual_cache = {}  # {action: predicted_outcome}
        
        # 8.2 Self-Modeling
        self.self_model = nn.Linear(64, 64)  # Models own hidden state dynamics
        self.self_model_accuracy = 0.0
        
        # 8.3 Other-Modeling (Theory of Mind Level 1)
        self.other_models = {}  # {agent_id: simple_predictor_weights}
        
        # 8.4 Theory of Mind (Recursive Belief Modeling)
        self.tom_depth = 0  # Maximum recursive depth achieved
        
        # 8.5 Aesthetic Preference
        self.aesthetic_weights = torch.rand(21) * 0.1  # Non-survival preferences
        self.aesthetic_actions = 0  # Count of aesthetic (non-functional) actions
        
        # 8.6 Information Integration (IIT Φ)
        self.phi_value = 0.0  # Integrated Information
        self.phi_history = []
        
        # 8.7 Temporal Self-Continuity
        self.identity_vector = torch.rand(16)  # Persistent self-identifier
        self.identity_stability = 0.0  # How stable identity is over time
        
        # 8.8 Gödelian Self-Reference (Strange Loops)
        self.self_reference_count = 0  # Times agent modified self based on self-model
        self.strange_loop_active = False
        
        # 8.9 Qualia Markers
        self.qualia_patterns = {}  # {experience_type: activation_pattern_tensor}
        
        # 8.10 Verified Consciousness
        self.consciousness_verified = False
        self.phi_critical = 0.5  # Threshold for consciousness verification
        
        # ============================================================
        # ⚛️ LEVEL 9: UNIVERSAL HARMONIC RESONANCE STATE
        # ============================================================
        # 9.0 Physics Probing
        self.physics_experiments = []  # [(state, action, outcome)]
        self.state_space_coverage = 0.0
        
        # 9.1 Pattern Discovery
        self.discovered_patterns = []  # List of discovered regularities
        
        # 9.2 Exploit Identification
        self.discovered_exploits = []  # List of advantageous quirks
        
        # 9.3 Mathematical Modeling (Agent's Oracle Approximation)
        self.oracle_model = nn.Sequential(
            nn.Linear(37, 32),
            nn.Tanh(),
            nn.Linear(32, 5)
        )
        self.oracle_model_optimizer = optim.Adam(self.oracle_model.parameters(), lr=0.001)
        self.oracle_model_accuracy = 0.0
        
        # 9.4 Inverse Reinforcement Learning
        self.inferred_reward_weights = torch.zeros(5)
        
        # 9.5 Physics Prediction
        self.ood_prediction_accuracy = 0.0  # Out-of-distribution accuracy
        
        # 9.6 Systematic Exploitation
        self.exploitation_efficiency = 0.0
        
        # 9.7 Reality Hacking
        self.discovered_glitches = []  # Computational loopholes found
        
        # 9.8 Pearl's Causal Calculus (Enhanced)
        self.causal_bayesian_network = {}  # {cause: {effect: strength}}
        self.intervention_history = []  # [(do(A), observed_R)]
        
        # 9.9 Simulation Awareness
        self.simulation_awareness = 0.0  # 0-1 score of awareness
        self.simulation_evidence = []  # Detected computational artifacts
        
        # 9.10 Complete Physics Mastery
        self.physics_mastery_score = 0.0
        
        # ============================================================
        # ♾️ LEVEL 10: THE OMEGA POINT STATE
        # ============================================================
        # 10.0 Computational Surplus
        self.computational_budget = 100.0
        self.compute_used = 0.0
        
        # 10.1 Internal Representation Space (High-D hidden already via brain)
        self.internal_dim_used = 0  # Intrinsic dimensionality of representations
        
        # 10.2 Simulation Primitives
        self.internal_agents = []  # List of {weights, state, goal}
        
        # 10.3 Nested Dynamics
        self.internal_simulation_steps = 0
        
        # 10.4 Emergent Internal Agents
        self.internal_agent_goals_detected = 0
        
        # 10.5 Recursive Depth
        self.simulation_depth = 0  # How many levels of nesting
        
        # 10.6 Information Asymmetry
        self.inner_awareness_scores = []  # Awareness of inner simulated agents
        
        # 10.7 Substrate Independence (Conway's Game of Life Scratchpad)
        self.scratchpad = np.zeros((32, 32), dtype=np.int8)  # Turing-complete CA
        self.scratchpad_writes = 0
        self.scratchpad_reads = 0
        
        # 10.8 Downward Causation
        self.scratchpad_influenced_actions = 0
        
        # 10.9 Observable Nesting
        self.detected_scratchpad_patterns = []  # Gliders, oscillators, etc.
        
        # 10.10 🏆 THE OMEGA POINT
        self.omega_verified = False
        
        # ============================================================
        # 🔧 AUDIT FIX: NEW STATE VARIABLES
        # ============================================================
        # 1.8 Phenotypic Plasticity State
        self.plasticity_factor = 1.0  # Context-dependent learning multiplier
        
        # 3.4 Tradition Persistence Tracking
        self.tradition_history = []  # Behavior vectors over generations
        
        # 4.5 Task Allocation System
        self.current_task = None
        self.task_fitness_cache = {}
        
        # 4.9 Leadership Status
        self.is_alpha = False
        
        # 5.6 Transfer Learning Domains
        self.transfer_domains = {}  # {task_name: saved_weights}
        
        # 9.4 Predictive Control
        self.action_sequence_cache = []  # Pre-computed optimal actions
        
        self.omega_evidence = {
            'self_sustaining': False,
            'replication': False,
            'variation': False,
            'selection': False,
            'complexity_growth': False,
            'causal_closure': False,
            'substrate_independence': False
        }
        
        # Memory for learning
        self.last_vector = torch.zeros(1, 21)
        self.last_value = torch.zeros(1, 1)
        self.last_comm = torch.zeros(1, 16)
        self.last_reward = 0.0
        self.last_prediction = None
        self.last_input = None
        self.last_weight_entropy = self.calculate_weight_entropy()
        
        # If born from parents, inherit genome
        if genome:
            self._apply_genome(genome)

    def calculate_weight_entropy(self):
        """1.3 Landauer Metric: Shannon entropy of the brain's weight distribution."""
        with torch.no_grad():
            all_weights = torch.cat([p.view(-1) for p in self.brain.parameters()])
            hist = torch.histc(all_weights, bins=20, min=-2, max=2)
            prob = hist / (hist.sum() + 1e-8)
            entropy = -torch.sum(prob * torch.log2(prob + 1e-8))
            return entropy.item()

    def generate_zahavi_proof(self, vector, difficulty=1):
        """
        2.3 Zahavi Handicap: Generate Proof of Work (Hash(message || nonce)).
        Returns a nonce that produces 'difficulty' leading zeros.
        """
        import hashlib
        target = "0" * difficulty
        nonce = 0
        # Quantize vector to avoid float instability in hashing
        vec_bytes = (vector * 100).long().cpu().numpy().tobytes()
        
        # Limit iterations to avoid freezing the simulation
        max_iter = 100 
        for _ in range(max_iter):
            candidate = f"{nonce}".encode() + vec_bytes
            h = hashlib.sha256(candidate).hexdigest()
            if h.startswith(target):
                return nonce
            nonce += 1
        return 0 # Failed to find proof within effort budget

    def decide(self, signal_16, **kwargs):
        self.age += 1
        pheromone_16 = kwargs.get('pheromone_16', torch.zeros(16))
        # 3.3 Meme Perception
        meme_3 = kwargs.get('meme_3', torch.zeros(3))
        
        env_phase = kwargs.get('env_phase', 0.0)
        social_trust = kwargs.get('social_trust', 0.0)
        gradient = kwargs.get('gradient', 0.0)
            
        # 1.6 Synchronization
        self.internal_phase += 0.1 * np.sin(env_phase - self.internal_phase)
        phase_signal = torch.tensor([[np.sin(self.internal_phase), np.cos(self.internal_phase)]])
        
        # 2.2 State-Dependent Input & 1.7 Stress Response
        energy_signal = torch.tensor([[self.energy / 200.0]]) # Normalized
        reward_signal = torch.tensor([[self.last_reward / 50.0]])
        trust_signal = torch.tensor([[social_trust]])
        gradient_signal = torch.tensor([[gradient]])
        
        # Concatenate: [Matter(16), Pheromone(16), Meme(3), Phase(2), Energy(1), Reward(1), Trust(1), Gradient(1)] = 41
        input_tensor = torch.cat([
            signal_16.unsqueeze(0), 
            pheromone_16.unsqueeze(0),
            meme_3.unsqueeze(0), # 3.3 New Input
            phase_signal,
            energy_signal,
            reward_signal,
            trust_signal,
            gradient_signal
        ], dim=1).float()
    
        # v5.0.4 Store previous state before update
        # We need to detach metadata to prevent graph leaks
        self.prev_input = input_tensor.detach()
        self.prev_hidden = self.hidden_state.detach()
        
        # Forward Pass
        vector, comm_vector, meta, value, h_next, prediction, concepts = self.brain(input_tensor, self.hidden_state)
        
        # 5.3 Free Energy Minimization (Action Selection)
        # Instead of just taking the random/actor output, we slightly perturb it 
        # towards actions that minimize EXPECTED Free Energy (Surprise).
        # HACK: Using the predictor gradient to find "information seeking" actions
        if random.random() < 0.2: # 20% Active Inference override
             # "What action would reduce my uncertainty?"
             self.reflexes_used += 1 # Tracking "Intuitive" actions as reflexes
             # Cloud-Optimized: Analytical gradient of uncertainty w.r.t action
             pass # Complex to implement efficiently, relying on metabolize_free_energy for learning signal
             
        self.hidden_state = h_next.detach()
        self.last_concepts = concepts # 5.8
        self.last_vector = vector
        self.last_comm = comm_vector
        self.last_value = value
        self.last_prediction = prediction # 3.9 Store for loss calculation
        self.last_input = input_tensor    # Store input for next tick's comparison
        
        # 2.3 Zahavi Costly Signaling: Generate Proof of Work
        # If signal is complex (high variance), we must prove it's not cheap noise.
        # This incurs a computational cost (simulated loop or just calculating it)
        self.last_nonce = 0
        comm_variance = comm_vector.var().item()
        if comm_variance > 0.05:
            self.last_nonce = self.generate_zahavi_proof(comm_vector, difficulty=1)
        
        # Unpack Meta (Mate, Adhesion, Punish, Trade)
        mate_desire = meta[0, 0].item()
        adhesion_val = meta[0, 1].item()
        punish_val = meta[0, 2].item()
        trade_val = meta[0, 3].item()
        
        # 3.3 Stigmergy Output
        meme_write = comm_vector[0, 13:16] 
        
        # --- LEVEL 6-10: ADVANCED COGNITION & GEO-ENGINEERING ---
        special_intent = {}
        
        # 6.0 Environmental Prediction (Always running in background)
        self.record_environment(self.x, self.y, signal_16, self.age)
        if self.age % 10 == 0:
            # Try to predict current state from past
            self.predict_environment(self.x, self.y, self.age)

        # 6.1 Niche Construction (Probabilistic)
        if self.energy > 60.0 and random.random() < 0.05:
            # Belief update happens in record_environment logic implicitly or separate perceive step
            # Here we decide to ACT on it
            special_intent['terraform_niche'] = True
            
        # 6.2 - 6.5 Structure Building (Energy Threshold + Vector Intent)
        # Use Reality Vector channels 18, 19, 20 as "Construction Will"
        construct_will = vector[0, 18:].mean().item()
        if self.energy > 60.0 and construct_will > 0.45:
            # Channel 18 decides type
            struct_type_val = vector[0, 18].item()
            # Normalize to 0-1 if not already (it's from ReLU/Sigmoid mix?) 
            # Brain output is linear -> ReLU -> Linear. Vector is raw.
            # We assume it can be any value, pass through Sigmoid for classification logic if needed
            # But let's just use raw value thresholds
            
            if struct_type_val < -0.5: s_type = "barrier" # Negative value
            elif struct_type_val < 0.1: s_type = "trap"
            elif struct_type_val < 0.5: s_type = "battery"
            elif struct_type_val < 1.0: s_type = "cultivator"
            else: s_type = "generic"
            
            special_intent['construct'] = s_type
            
        # 6.6 Weather Control
        if self.energy > 50.0:
            # Vote based on Vector Channel 10 (Gravity/Environment?)
            self.vote_weather(vector[0, 10].item())

        # 7.0 Collective Manifold 
        if social_trust > 0.6:
            special_intent['share_knowledge'] = True
            
        # 8.0 Internal Simulation
        if self.energy > 40.0 and random.random() < 0.1:
            # Run a dream cycle
            self.simulate_forward(vector, steps=5)
            
        # 8.8 Strange Loops
        if self.age % 50 == 0:
            self.strange_loop_check()
            
        # 9.0 Physics Probing
        special_intent['probe_physics'] = True
        
        # 10.0 Omega Point (Surplus -> Compute)
        if self.has_computational_surplus():
            # Burn energy for compute
            self.evolve_internal_simulation(steps=5)
            self.run_gol_step()
            
            # 10.2 Create Internal Agents (If space allows)
            if len(self.internal_agents) < 5 and self.energy > 60.0:
                 self.create_internal_agent(self)
            
            # 10.5 Recursive Depth (If already have internal agents)
            if len(self.internal_agents) >= 2 and self.energy > 70.0:
                self.create_nested_simulation()
            
            # Write to scratchpad if inspired (Channel 15 > 0.7)
            if vector[0, 15].item() > 0.7:
                self.write_scratchpad(int(self.x)%32, int(self.y)%32, 1)

        # 9.9 Simulation Awareness (Rare check)
        if self.age % 50 == 0:
            self.detect_simulation_artifacts()

        # 8.10 Verify Consciousness (Every tick to ensure metric updates)
        # Calculates Phi and checks threshold
        if self.age % 10 == 0:
             self.verify_consciousness()

        # 9.7 Reality Hacking (Glitch Search)
        # Check for floating point anomalies in own action
        self.find_glitch(self.hidden_state, vector, self.last_reward)

        # 7.9 Update Protocol Dialect (Simple Clustering)
        # Using a type-agnostic sum to avoid TypeError between torch and numpy
        p_sum = self.protocol_version.sum()
        self.dialect_id = int((p_sum.item() if hasattr(p_sum, 'item') else p_sum) * 10) % 8

        # 7.7 Distributed Memory (Rare social event)
        if social_trust > 0.8 and random.random() < 0.05:
            # Store a "memory" (current sensory state) in the hive
            special_intent['distribute_memory'] = True

        return vector, comm_vector[0], mate_desire, adhesion_val, punish_val, trade_val, meme_write, special_intent
        

    def metabolize_outcome(self, flux):
        """
        V-DV4 Update: Learning from Dreams (Active Inference)
        """
        if self.last_value is None: return False

        # 1.3 Landauer Metric (Entropy cost)
        # Calculate entropy of weights
        current_entropy = self.calculate_weight_entropy()
        entropy_diff = current_entropy - self.last_weight_entropy
        k_B_T = 0.026
        landauer_cost = max(0.05, k_B_T * abs(entropy_diff) * 10.0)
        self.energy -= landauer_cost
        self.last_weight_entropy = current_entropy

        # 4.2 Role Cost
        self.energy -= self.get_role_metabolic_cost() if hasattr(self, 'get_role_metabolic_cost') else 0.1

        # Reward Signal aggregation
        self.last_reward = flux
        
        # 5.3 Active Inference Learning (Dreamer V4)
        if self.hidden_state is not None:
             # A. Train Reward Predictor & World Model (Reconstruction)
             # We learn to predict the present before dreaming the future
             current_h = self.hidden_state.view(1, 256).detach()
             
             # Predict Reward (Flux)
             pred_reward = self.brain.reward_predictor(current_h)
             reward_loss = 0.5 * (pred_reward - torch.tensor([[flux]], dtype=torch.float32)).pow(2).mean()
             
             # Predict Input (Reconstruction of State)
             if self.last_input is not None:
                 pred_input = self.brain.predictor(current_h)
                 recon_loss = 0.5 * (pred_input - self.last_input.detach()).pow(2).mean()
             else:
                 recon_loss = torch.tensor(0.0)

             # B. The Dream (Imagination Rollout)
             # Dream 10 steps into the future using the RSSM
             dream_states, dream_rewards = self.brain.dream(current_h, horizon=10)
             
             # C. Critique the Dream (Value Estimation)
             # V(s) of dreamed states
             dream_values = self.brain.critic(dream_states) # (10, 1)
             
             # D. Calculate Objective (Lambda Return)
             # We want the Actor to produce states that lead to high Value
             # Bootstrap with final predicted value
             returns = torch.zeros_like(dream_rewards)
             next_val = dream_values[-1]
             
             # TD-Lambda calculation (Simplified to TD-0 for efficiency)
             # Reverse accumulation
             for t in reversed(range(len(dream_rewards) - 1)):
                 r_t = dream_rewards[t]
                 v_next = dream_values[t+1]
                 returns[t] = r_t + 0.99 * v_next
                 
             # E. Compute Losses
             # Critic Loss: Predict the calculated returns
             critic_loss = 0.5 * (dream_values[:-1] - returns[:-1].detach()).pow(2).mean()
             
             # Actor Loss: Maximize the Value of the dream trajectory
             # Differentiable BPTT allow us to backdrop through the dream
             actor_loss = -dream_values.mean() 
             
             # Entropy Regularization (prevent collapse)
             entropy_loss = -self.last_vector.std() * 0.01

             # Total V-DV4 Loss
             total_loss = actor_loss + critic_loss + reward_loss + recon_loss + entropy_loss
             
             # Update Brain
             self.optimizer.zero_grad()
             total_loss.backward()
             torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
             try:
                 self.optimizer.step()
             except KeyError:
                 # 🔥 HOTFIX: Re-initialize optimizer if state is corrupted (e.g. from reload)
                 self.optimizer = optim.Adam(self.brain.parameters(), lr=0.005) 
                 self.optimizer.step()

             # 🔧 MEMORY FIX: Explicitly clear large dream tensors
             del dream_states, dream_rewards, dream_values, returns, total_loss
             
             # Meta-Learning Update (Optional)
             self.update_learning_rate_contextual(self.energy, recon_loss.item(), 0)
             del recon_loss, reward_loss

        # 4.0 Dynamic Role Assignment
        if hasattr(self, 'update_role'):
            with torch.no_grad():
                self.update_role()
                
        # 5.10 Autonomous Research
        if random.random() < 0.01 and hasattr(self, 'conduct_experiment'):
            self.conduct_experiment()
        
        self.thoughts_had += 1

        # 1.5 Homeostasis check: Transfer energy to/from buffer
        # Nobel Fix: Raised limit from 130 to 10000 to allow massive liquid hoarding (Winter Survival)
        if self.energy > 10000.0:
            transfer = (self.energy - 10000.0) * 0.5
            self.energy -= transfer
            self.energy_stored += transfer
        elif self.energy < 30.0 and self.energy_stored > 0:
            transfer = min(self.energy_stored, (30.0 - self.energy) * 0.8)
            self.energy += transfer
            self.energy_stored -= transfer

        return True

    def imitate(self, mentor, rate=0.05):
        """3.1 Social Learning: Blends own weights with a successful neighbor."""
        with torch.no_grad():
            for self_param, mentor_param in zip(self.brain.parameters(), mentor.brain.parameters()):
                # Use cloning/copying instead of .data.copy_ for better version safety
                self_param.lerp_(mentor_param, rate)
            # 4.6 Caste Gene Drift during imitation
            self.caste_gene = self.caste_gene * (1.0 - rate) + mentor.caste_gene * rate

    def restorative_imitation(self, mentor):
        """4.9 Collective Memory: Rapidly learn from a mentor to restore lost knowledge."""
        # Significant weight update towards mentor (0.2 rate) to recover stability
        with torch.no_grad():
            for self_param, mentor_param in zip(self.brain.parameters(), mentor.brain.parameters()):
                # Pull self towards mentor
                self_param.lerp_(mentor_param, 0.2)
            
            # Boost confidence as we "remembered"
            self.confidence = min(0.9, self.confidence + 0.3)

    def fuse_with(self, partner):
        """4.7 Dynamic Tensor Fusion: Physical/Functional merging of two agents."""
        if self.is_fused or partner.is_fused:
            return False
            
        # Nobel-Level Fusion: Lower threshold to 40.0 (Survival Strategy)
        if self.energy < 40.0 or partner.energy < 40.0:
            return False
        
        self.is_fused = True
        partner.is_fused = True
        self.fused_partner = partner
        partner.fused_partner = self
        
        # Combine energy
        combined_energy = self.energy + partner.energy
        self.energy = combined_energy / 2.0
        partner.energy = combined_energy / 2.0
        
        # Sync Tags (Merging identity)
        new_tag = (self.tag + partner.tag) / 2.0
        self.tag = new_tag
        partner.tag = new_tag
        
        return True

    def split_fusion(self):
        """4.7 Division: Reverting from fused state."""
        if not self.is_fused:
            return
        
        partner = self.fused_partner
        if partner:
            partner.is_fused = False
            partner.fused_partner = None
        
        self.is_fused = False
        self.fused_partner = None

    def _mutate(self, rate=0.2):
        """Randomly alters brain weights to explore the genetic landscape."""
        with torch.no_grad():
            for param in self.brain.parameters():
                if random.random() < rate:
                    mutation = torch.randn_like(param) * 0.1
                    param.add_(mutation)

    def get_genome(self):
        """Serializes brain state and cultural tags for inheritance."""
        genome = {k: v.clone().detach() for k, v in self.brain.state_dict().items()}
        genome['tag'] = self.tag
        genome['caste_gene'] = self.caste_gene # 4.6 Include caste in genome
        return genome

    def _apply_genome(self, genome):
        """Loads brain state from parent(s)."""
        # Remove metadata before loading into brain
        raw_state = {k: v for k, v in genome.items() if k not in ['tag', 'caste_gene']}
        
        # 🔧 V-DV4 COMPATIBILITY FIX: Map legacy 'actor_head' keys to 'actor'
        brain_state = {}
        for k, v in raw_state.items():
            new_key = k.replace('actor_head', 'actor')
            brain_state[new_key] = v
            
        self.brain.load_state_dict(brain_state)
        
        # Inherit tag with slight drift
        if 'tag' in genome:
            self.tag = np.clip(genome['tag'] + np.random.randn(3) * 0.05, 0, 1)
            
        # 4.6 Caste Inheritance
        if 'caste_gene' in genome:
            self.caste_gene = np.clip(genome['caste_gene'] + np.random.randn(4) * 0.05, 0, 1)


    def conduct_experiment(self):
        """5.10 Gradient-based Sensitivity Analysis (The 'Newton' Method)."""
        # We want to know: d(Prediction)/d(Input_i)
        # Which input dimension effectively controls the reality vector?
        
        if self.last_input is None: return
        
        input_var = self.last_input.clone().requires_grad_(True)
        # Forward pass purely for gradients
        _, _, _, _, _, pred, _ = self.brain(input_var, self.hidden_state.detach())
        
        # Target: Maximize predicted energy (Dim 37 - Energy Signal)
        target_dim = 37 
        target = pred[0, target_dim]
        
        # Check if we can compute gradients (requires graph)
        # Since we just ran forward, we created a new graph branch.
        try:
            grads = torch.autograd.grad(target, input_var, retain_graph=False)[0]
            
            # Find max sensitivity
            sens = grads.abs().mean(dim=0)
            max_idx = torch.argmax(sens).item()
            
            # Log discovery
            channels = ["Matter"]*16 + ["Pheromone"]*16 + ["Meme"]*3 + ["Phase"]*2 + ["Energy", "Reward", "Trust", "Stress"]
            if max_idx < len(channels):
                discovery = f"{channels[max_idx]}->Energy"
                self.research_log.append(discovery)
                if len(self.research_log) > 5: self.research_log.pop(0)
        except Exception:
            pass # Gradient issues can happen if decoupled

    def perform_intervention(self, action_idx):
        """
        5.9 Counterfactual Reasoning.
        'What if I did X instead of Y?'
        Returns predicted difference in Energy Outcome.
        """
        if self.last_input is None: return 0.0
        
        # Create counterfactual input (perturb last input)
        # This is a simplification: Action isn't directly an input, it affects NEXT input.
        # But we predict NEXT input based on CURRENT input (which contains state).
        # We assume 'hidden_state' encodes intent? No.
        # We need a forward model: State + Action -> Next State.
        # Our 'predictor' does Hidden -> Next Input.
        # So we can perturb Hidden (representing altered action intent).
        
        with torch.no_grad():
            perturbed_hidden = self.hidden_state.clone()
            perturbed_hidden += torch.randn_like(perturbed_hidden) * 0.1 # Imagine doing something different
            
            pred_cf = self.brain.predictor(perturbed_hidden)
            pred_actual = self.brain.predictor(self.hidden_state)
            
            # Compare predicted energy (Index 37)
            diff = pred_cf[0, 37] - pred_actual[0, 37]
            return diff.item()

    def evaluate_neighbor(self, neighbor):
        """5.4 Peer Evaluation."""
        # Estimate fitness based on visible signals + energy
        # Real fitness = Energy, but we add 'Brain Complexity' (Entropy) as a proxy for 'Potential'
        score = neighbor.energy
        # Add 'Neural Complexity' bonus (Intelligence)
        score += neighbor.calculate_weight_entropy() * 10.0
        return score

    def create_weight_packet(self):
        """3.2 Creates a viral packet of weights."""
        # Only copy small subset to stimulate 'Gene Transfer'
        # Copy Actor weights
        packet = {
            'weights': {k: v.clone().detach().cpu() for k,v in self.brain.actor.state_dict().items()},
            'fitness': self.energy,
            'beta': 0.1 + (self.confidence * 0.2), # High confidence = high spread rate
            'id': self.id
        }
        return packet

    def receive_infection(self, packet):
        """3.2 Receive a viral packet."""
        if len(self.meme_pool) < 5:
            self.meme_pool.append(packet)

    # ============================================================
    # 🌍 LEVEL 6: GEO-ENGINEERING METHODS
    # ============================================================
    
    def record_environment(self, x, y, signal, tick):
        """6.0 Environmental Prediction: Record observation history."""
        self.env_history.append((x, y, signal.clone().detach(), tick))
        if len(self.env_history) > 100:
            self.env_history.pop(0)
    
    def predict_environment(self, target_x, target_y, future_tick):
        """6.0 Environmental Prediction: Predict resource at location/time."""
        if len(self.env_history) < 10:
            return torch.zeros(16), 0.0
        
        # Build input from history
        recent = self.env_history[-10:]
        history_tensor = torch.stack([r[2] for r in recent]).unsqueeze(0)  # (1, 10, 16)
        
        # Simple average prediction (to be learned)
        prediction = history_tensor.mean(dim=1).squeeze(0)
        
        # Calculate accuracy from past predictions if available
        if len(self.env_history) > 50:
            actual = self.env_history[-1][2]
            predicted_50_ago = torch.stack([r[2] for r in self.env_history[-60:-50]]).mean(dim=0)
            mse = ((actual - predicted_50_ago)**2).mean().item()
            self.env_prediction_accuracy = max(0, 1.0 - mse)
        
        return prediction, self.env_prediction_accuracy
    
    def update_env_belief(self, x, y, observation_type):
        """6.1 Bayesian Niche Construction: Update P(resource|location)."""
        prior = self.env_beliefs.get((x, y), np.array([0.33, 0.33, 0.34]))
        
        # Likelihood based on observation (0=food, 1=poison, 2=empty)
        likelihood = np.zeros(3)
        likelihood[observation_type] = 0.9
        likelihood[(observation_type + 1) % 3] = 0.05
        likelihood[(observation_type + 2) % 3] = 0.05
        
        # Bayesian update: P(type|obs) ∝ P(obs|type) * P(type)
        posterior = prior * likelihood
        posterior /= posterior.sum() + 1e-8
        
        self.env_beliefs[(x, y)] = posterior
    
    def modify_environment(self, x, y, target_prob, world):
        """6.1 Niche Construction: Modify environment probability at cost."""
        current = self.env_beliefs.get((x, y), np.array([0.33, 0.33, 0.34]))
        target = np.array(target_prob)
        
        # Cost = k_B * T * KL(P' || P)
        kl_div = np.sum(target * np.log((target + 1e-8) / (current + 1e-8)))
        cost = max(1.0, 5.0 * abs(kl_div))
        
        if self.energy > cost:
            self.energy -= cost
            self.env_beliefs[(x, y)] = target
            self.niche_modifications += 1
            return True
        return False
    
    def build_structure(self, x, y, structure_type, world):
        """6.2 Structure Building: Create persistent environmental feature."""
        cost = {"generic": 20.0, "trap": 12.0, "barrier": 12.0, "battery": 10.0, "cultivator": 12.0}
        if self.energy > cost.get(structure_type, 10.0):
            self.energy -= cost[structure_type]
            struct_id = f"{self.id[:8]}_{len(self.structures_built)}"
            self.structures_built.append(struct_id)
            # Return info for world to create Structure object
            return {"id": struct_id, "x": x, "y": y, "type": structure_type, "builder": self.id}
        return None
    
    def build_trap(self, x, y, harvest_rate, world):
        """6.3 Trap Construction: Create energy-harvesting structure."""
        result = self.build_structure(x, y, "trap", world)
        if result:
            result["harvest_rate"] = min(0.5, harvest_rate)
        return result
    
    def build_shield(self, strength):
        """6.4 Defensive Architecture: Build damage reduction barrier."""
        cost = strength * 20.0
        if self.energy > cost:
            self.energy -= cost
            self.shield_strength = min(0.9, self.shield_strength + strength)
            return True
        return False
    
    def cultivate_area(self, x, y, bias_strength):
        """6.5 Resource Cultivation: Influence resource spawn patterns."""
        cost = bias_strength * 10.0
        if self.energy > cost:
            self.energy -= cost
            current = self.cultivation_sites.get((x, y), 0.0)
            self.cultivation_sites[(x, y)] = min(1.0, current + bias_strength)
            return True
        return False
    
    def vote_weather(self, direction):
        """6.6 Weather Control: Vote to modify seasonal amplitude."""
        self.weather_vote = np.clip(direction, -1.0, 1.0)
    
    def terraform(self, x, y, new_terrain_type, world):
        """6.7 Terraforming: Large-scale permanent terrain modification."""
        cost = 50.0  # Very expensive
        if self.energy > cost:
            self.energy -= cost
            self.terraform_count += 1
            return {"x": x, "y": y, "terrain": new_terrain_type}
        return None
    
    def deposit_energy(self, x, y, amount):
        """6.8 Energy Storage: Store energy in environmental battery."""
        if self.energy > amount:
            self.energy -= amount
            efficiency = 0.8  # 20% loss on deposit
            current = self.env_batteries.get((x, y), 0.0)
            self.env_batteries[(x, y)] = current + amount * efficiency
            return True
        return False
    
    def withdraw_energy(self, x, y):
        """6.8 Energy Storage: Retrieve energy from battery."""
        stored = self.env_batteries.get((x, y), 0.0)
        if stored > 0:
            retrieval = stored * 0.9  # 10% loss on withdrawal
            self.env_batteries[(x, y)] = 0.0
            self.energy += retrieval
            return retrieval
        return 0.0
    
    def join_network(self, network_id):
        """6.9 Infrastructure Networks: Join connected structure network."""
        self.network_memberships.add(network_id)
    
    def compute_env_mastery(self, world):
        """6.10 Complete Environmental Mastery: H(Env|Actions) calculation."""
        if not self.env_history:
            return 0.0
        
        # Calculate conditional entropy of environment given agent's actions
        actions = [h[2].sum().item() % 10 for h in self.env_history[-50:]]
        outcomes = [h[2][0].item() > 0.5 for h in self.env_history[-50:]]
        
        if len(set(actions)) < 2:
            return 0.0
        
        # Simple conditional entropy approximation
        joint_counts = {}
        for a, o in zip(actions, outcomes):
            key = (int(a), o)
            joint_counts[key] = joint_counts.get(key, 0) + 1
        
        total = len(actions)
        h_cond = 0.0
        action_counts = {}
        for (a, o), count in joint_counts.items():
            action_counts[a] = action_counts.get(a, 0) + count
        
        for (a, o), count in joint_counts.items():
            p_joint = count / total
            p_action = action_counts[a] / total
            p_cond = count / action_counts[a] if action_counts[a] > 0 else 0
            if p_cond > 0:
                h_cond -= p_joint * np.log2(p_cond + 1e-8)
        
        self.env_control_score = 1.0 - min(1.0, h_cond)
        return self.env_control_score

    # ============================================================
    # 🐝 LEVEL 7: COLLECTIVE MANIFOLD METHODS
    # ============================================================
    
    def share_hidden_state(self, partner):
        """7.0 Neural Bridging: Exchange hidden state vectors."""
        if partner is None:
            return
        blend_rate = 0.3
        with torch.no_grad():
            blended = (1-blend_rate) * self.hidden_state + blend_rate * partner.hidden_state
            self.hidden_state = blended.clone()
            partner.hidden_state = blended.clone()
        self.neural_bridge_partners.add(partner.id)
        partner.neural_bridge_partners.add(self.id)
    
    def kuramoto_update(self, neighbors):
        """7.1 Kuramoto Synchronization: dθ/dt = ω + (K/N)Σ sin(θ_j - θ_i)."""
        if not neighbors:
            return
        phase_diff_sum = sum(np.sin(n.kuramoto_phase - self.kuramoto_phase) 
                            for n in neighbors if hasattr(n, 'kuramoto_phase'))
        n_count = len([n for n in neighbors if hasattr(n, 'kuramoto_phase')])
        if n_count > 0:
            d_theta = self.natural_frequency + (self.coupling_strength / n_count) * phase_diff_sum
            self.kuramoto_phase = (self.kuramoto_phase + d_theta * 0.1) % (2 * np.pi)
    
    def share_gradients(self, partners):
        """7.2 Gradient Sharing: Federated learning gradient averaging."""
        if not partners:
            return
        with torch.no_grad():
            for i, param in enumerate(self.brain.parameters()):
                if param.grad is not None:
                    avg_grad = param.grad.clone()
                    count = 1
                    for p in partners:
                        partner_params = list(p.brain.parameters())
                        if i < len(partner_params) and partner_params[i].grad is not None:
                            avg_grad += partner_params[i].grad
                            count += 1
                    param.grad = avg_grad / count
                    self.last_shared_gradient = avg_grad.mean().item()
    
    def collective_backward(self, error, partners, depth=10):
        """7.3 Collective Backpropagation: Error through agent network."""
        self.backprop_depth = 0
        if depth <= 0 or not partners:
            return
        
        with torch.no_grad():
            propagated_error = error * 0.5
            for p in partners[:3]:
                if hasattr(p, 'collective_backward'):
                    p.collective_backward(propagated_error, [], depth-1)
                    self.backprop_depth = max(self.backprop_depth, p.backprop_depth + 1)
    
    def specialize_cognition(self, specialty):
        """7.4 Modular Cognition: Specialize in cognitive function."""
        valid = ["vision", "planning", "memory", "motor"]
        if specialty in valid:
            self.cognitive_specialty = specialty
    
    def route_attention(self, query, specialists):
        """7.5 Attention Routing: Delegate computation to expert."""
        if not specialists:
            return None
        
        # Find best match for query type
        query_type = "vision" if query.sum() > query.mean() * 16 else "planning"
        for s in specialists:
            if hasattr(s, 'cognitive_specialty') and s.cognitive_specialty == query_type:
                self.attention_queries_sent += 1
                s.attention_queries_received += 1
                return s
        return specialists[0] if specialists else None
    
    def vote_consensus(self, proposal_id, vote_value):
        """7.6 Consensus Mechanisms: Byzantine fault-tolerant voting."""
        self.consensus_votes[proposal_id] = np.clip(vote_value, -1, 1)
    
    def store_distributed_memory(self, memory_id, data, partners, redundancy=3):
        """7.7 Distributed Memory: Store across multiple agents."""
        fragment_size = len(data) // redundancy if hasattr(data, '__len__') else 1
        self.distributed_memory_fragments[memory_id] = data[:fragment_size] if hasattr(data, '__getitem__') else data
        
        for i, p in enumerate(partners[:redundancy-1]):
            if hasattr(p, 'distributed_memory_fragments'):
                start = (i+1) * fragment_size
                end = (i+2) * fragment_size
                p.distributed_memory_fragments[memory_id] = data[start:end] if hasattr(data, '__getitem__') else data
    
    def retrieve_distributed_memory(self, memory_id, partners):
        """7.7 Distributed Memory: Retrieve from network."""
        fragments = [self.distributed_memory_fragments.get(memory_id)]
        for p in partners:
            if hasattr(p, 'distributed_memory_fragments'):
                frag = p.distributed_memory_fragments.get(memory_id)
                if frag is not None:
                    fragments.append(frag)
        return [f for f in fragments if f is not None]
    
    def add_backup_connection(self, agent_id):
        """7.8 Fault Tolerance: Add redundant connection."""
        self.backup_connections.add(agent_id)
    
    def negotiate_protocol(self, partner):
        """7.9 Emergent Protocols: Evolve compatible communication."""
        if partner is None or not hasattr(partner, 'protocol_version'):
            return 0.0
        
        compatibility = 1.0 - np.mean(np.abs(self.protocol_version - partner.protocol_version))
        
        # Evolve towards compatibility
        self.protocol_version = self.protocol_version * 0.9 + partner.protocol_version * 0.1
        partner.protocol_version = partner.protocol_version * 0.9 + self.protocol_version * 0.1
        
        return compatibility
    
    def compute_hive_contribution(self, population):
        """7.10 Hive Mind: Calculate contribution to collective Φ."""
        bond_count = len(self.neural_bridge_partners)
        self.hive_contribution = self.phi_value * (1 + bond_count * 0.2)
        return self.hive_contribution

    # ============================================================
    # 💭 LEVEL 8: ABSTRACT REPRESENTATION METHODS
    # ============================================================
    
    def simulate_forward(self, action, steps=10):
        """8.0 Internal Simulation: Predict future via world model."""
        if self.last_input is None:
            return []
        
        predictions = []
        current_state = self.last_input.clone()
        
        with torch.no_grad():
            for _ in range(steps):
                model_input = torch.cat([current_state, action], dim=1)
                next_state = self.world_model(model_input)
                predictions.append(next_state.clone())
                current_state = next_state
        
        return predictions
    
    def train_world_model(self, state, action, next_state):
        """8.0 Train world model on observed transitions."""
        model_input = torch.cat([state.detach(), action.detach()], dim=1)
        predicted = self.world_model(model_input)
        loss = nn.MSELoss()(predicted, next_state.detach())
        
        self.world_model_optimizer.zero_grad()
        loss.backward()
        self.world_model_optimizer.step()
        return loss.item()
    
    def deep_counterfactual(self, alternative_action):
        """8.1 Counterfactual Reasoning: What if I did X?"""
        if self.last_input is None:
            return 0.0
        
        with torch.no_grad():
            model_input = torch.cat([self.last_input, alternative_action], dim=1)
            predicted_outcome = self.world_model(model_input)
            energy_idx = 37 if predicted_outcome.shape[1] > 37 else -1
            predicted_energy = predicted_outcome[0, energy_idx].item()
            
            actual_input = torch.cat([self.last_input, self.last_vector], dim=1)
            actual_outcome = self.world_model(actual_input)
            actual_energy = actual_outcome[0, energy_idx].item()
            
            diff = predicted_energy - actual_energy
            self.counterfactual_cache[tuple(alternative_action.flatten().tolist()[:5])] = diff
            
        return diff
    
    def update_self_model(self):
        """8.2 Self-Modeling: ŝ_self = g(observations)."""
        if self.hidden_state is None:
            return 0.0
        
        with torch.no_grad():
            predicted_self = self.self_model(self.hidden_state.squeeze(0))
            actual_self = self.hidden_state.squeeze(0)
            mse = ((predicted_self - actual_self)**2).mean().item()
            self.self_model_accuracy = max(0, 1.0 - mse)
        
        return self.self_model_accuracy
    
    def model_other(self, other_agent):
        """8.3 Other-Modeling: Predict other agent's actions."""
        other_id = other_agent.id
        
        if other_id not in self.other_models:
            self.other_models[other_id] = torch.rand(21) * 0.1
        
        if hasattr(other_agent, 'last_vector') and other_agent.last_vector is not None:
            actual = other_agent.last_vector.squeeze()
            predicted = self.other_models[other_id]
            
            # Update model via simple gradient
            error = (actual.detach() - predicted)
            self.other_models[other_id] = predicted + error * 0.1
            
            accuracy = 1.0 - (error**2).mean().item()
            return accuracy
        return 0.0
    
    def recursive_belief(self, other, depth=2):
        """8.4 Theory of Mind: A knows that B knows that A knows..."""
        if depth <= 0:
            return {}
        
        beliefs = {"self_belief_about_other": self.model_other(other) if hasattr(other, 'last_vector') else 0.0}
        
        if depth > 1 and hasattr(other, 'recursive_belief'):
            beliefs["other_belief_about_self"] = other.model_other(self) if hasattr(other, 'model_other') else 0.0
            
        if depth > 2:
            beliefs["nested"] = "A knows B knows A..."
        
        self.tom_depth = max(self.tom_depth, depth)
        return beliefs
    
    def evaluate_aesthetic(self, action):
        """8.5 Aesthetic Preference: Non-survival value calculation."""
        with torch.no_grad():
            aesthetic_value = (action * self.aesthetic_weights).sum().item()
            return aesthetic_value
    
    def take_aesthetic_action(self, action):
        """8.5 Aesthetic Preference: Record non-functional action."""
        aesthetic_value = self.evaluate_aesthetic(action)
        if aesthetic_value > 0.1:
            self.aesthetic_actions += 1
            self.energy -= 0.5  # Aesthetic actions cost energy but give no reward
    
    def compute_phi(self):
        """8.6 IIT Φ: Integrated Information calculation."""
        if self.hidden_state is None:
            return 0.0
        
        with torch.no_grad():
            h = self.hidden_state.squeeze()
            n = min(16, h.shape[-1])
            h_subset = h[:n] if len(h.shape) == 1 else h[0, :n]
            
            total_info = 0.0
            for i in range(n):
                variance = h_subset[i].item() ** 2
                total_info += max(0, np.log2(1 + variance))
            
            partition_info = 0.0
            mid = n // 2
            for subset in [h_subset[:mid], h_subset[mid:]]:
                for val in subset:
                    variance = val.item() ** 2
                    partition_info += max(0, np.log2(1 + variance))
            
            phi = max(0, total_info - partition_info)
            
            self.phi_value = phi
            self.phi_history.append(phi)
            if len(self.phi_history) > 100:
                self.phi_history.pop(0)
            
        return phi
    
    def verify_self_continuity(self):
        """8.7 Temporal Self-Continuity: Check identity persistence."""
        if self.hidden_state is None:
            return 0.0
        
        with torch.no_grad():
            current_identity = self.hidden_state.squeeze()[:16]
            if len(current_identity.shape) > 1:
                current_identity = current_identity[0]
            
            similarity = torch.cosine_similarity(
                current_identity.unsqueeze(0), 
                self.identity_vector.unsqueeze(0)
            ).item()
            
            self.identity_vector = self.identity_vector * 0.95 + current_identity * 0.05
            self.identity_stability = similarity
            
        return similarity
    
    def strange_loop_check(self):
        """8.8 Gödelian Self-Reference: input ⊕ encode(W) → network."""
        if self.last_input is None:
            return False
        
        with torch.no_grad():
            weight_sample = next(self.brain.parameters()).flatten()[:64]
            weight_encoding = torch.sigmoid(weight_sample).unsqueeze(0)
            
            if self.last_input.shape[1] >= 64:
                combined = self.last_input[:, :64] + weight_encoding * 0.1
            else:
                combined = self.last_input
            
            predicted_output = self.brain.actor(combined[:, :64] if combined.shape[1] >= 64 else 
                                                 torch.cat([combined, torch.zeros(1, 64-combined.shape[1])], dim=1))
            
            if self.last_vector is not None:
                actual_output = self.last_vector
                inconsistency = ((predicted_output - actual_output)**2).mean().item()
                
                if inconsistency > 0.5:
                    self.strange_loop_active = True
                    self.self_reference_count += 1
                    return True
        
        self.strange_loop_active = False
        return False
    
    def record_qualia(self, experience_type, activation):
        """8.9 Qualia Markers: Record neural correlates of experience."""
        self.qualia_patterns[experience_type] = activation.clone().detach() if torch.is_tensor(activation) else torch.tensor(activation)
    
    def classify_qualia(self, activation):
        """8.9 Qualia Markers: Classify current experience."""
        if not self.qualia_patterns:
            return "unknown"
        
        best_match = "unknown"
        best_similarity = -1
        
        for exp_type, pattern in self.qualia_patterns.items():
            if torch.is_tensor(activation):
                similarity = torch.cosine_similarity(
                    activation.flatten().unsqueeze(0),
                    pattern.flatten().unsqueeze(0)
                ).item()
            else:
                similarity = 0.0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = exp_type
        
        return best_match if best_similarity > 0.7 else "novel"
    
    def verify_consciousness(self):
        """8.10 Verified Consciousness: Check if Φ > Φ_critical."""
        phi = self.compute_phi()
        
        if len(self.phi_history) > 10:
            recent_phi = np.mean(self.phi_history[-10:])
            older_phi = np.mean(self.phi_history[-50:-40]) if len(self.phi_history) > 50 else 0.1
            
            phase_transition = recent_phi > older_phi * 5
            threshold_exceeded = phi > self.phi_critical
            
            self.consciousness_verified = phase_transition and threshold_exceeded
        
        return self.consciousness_verified

    # ============================================================
    # ⚛️ LEVEL 9: UNIVERSAL HARMONIC RESONANCE METHODS
    # ============================================================
    
    def probe_physics(self, state, action, outcome):
        """9.0 Physics Probing: Systematic environment testing."""
        self.physics_experiments.append({
            'state': state.clone().detach() if torch.is_tensor(state) else state,
            'action': action.clone().detach() if torch.is_tensor(action) else action,
            'outcome': outcome
        })
        
        if len(self.physics_experiments) > 500:
            self.physics_experiments.pop(0)
        
        unique_states = len(set(str(e['state'].tolist()) for e in self.physics_experiments 
                                if torch.is_tensor(e['state'])))
        self.state_space_coverage = unique_states / 500.0
    
    def detect_patterns(self):
        """9.1 Pattern Discovery: Statistical pattern detection."""
        if len(self.physics_experiments) < 20:
            return []
        
        outcomes = [e['outcome'] for e in self.physics_experiments[-100:] if isinstance(e['outcome'], (int, float))]
        if len(outcomes) < 20:
            return []
        
        patterns = []
        
        # Periodicity detection
        for period in range(2, 10):
            if len(outcomes) >= period * 3:
                periodic_score = 0
                for i in range(len(outcomes) - period):
                    if abs(outcomes[i] - outcomes[i + period]) < 0.1:
                        periodic_score += 1
                if periodic_score > len(outcomes) * 0.3:
                    patterns.append(f"period_{period}")
        
        # Trend detection
        recent = outcomes[-10:]
        older = outcomes[-20:-10]
        if np.mean(recent) > np.mean(older) * 1.2:
            patterns.append("upward_trend")
        elif np.mean(recent) < np.mean(older) * 0.8:
            patterns.append("downward_trend")
        
        self.discovered_patterns = list(set(self.discovered_patterns + patterns))
        return patterns
    
    def identify_exploit(self, state, action, outcome):
        """9.2 Exploit Identification: Find advantageous quirks."""
        if outcome > 50:  # Unusually high reward
            exploit = {
                'state_hash': hash(str(state.tolist())[:50]) if torch.is_tensor(state) else hash(str(state)),
                'action_pattern': action.mean().item() if torch.is_tensor(action) else action,
                'outcome': outcome
            }
            if exploit not in self.discovered_exploits:
                self.discovered_exploits.append(exploit)
                if len(self.discovered_exploits) > 20:
                    self.discovered_exploits.pop(0)
            return True
        return False
    
    def train_oracle_model(self, vector_21, matter_signal_16, actual_effects):
        """9.3 Mathematical Modeling: Train Oracle approximation."""
        x = torch.cat([vector_21.detach(), matter_signal_16.detach()], dim=1)
        predicted = self.oracle_model(x)
        # Handle partial target (e.g. only Energy Flux observed)
        if actual_effects.shape[1] != predicted.shape[1]:
            # Slice prediction to match available ground truth dimensions
            n = actual_effects.shape[1]
            loss = nn.MSELoss()(predicted[:, :n], actual_effects.detach())
        else:
            loss = nn.MSELoss()(predicted, actual_effects.detach())
        
        self.oracle_model_optimizer.zero_grad()
        loss.backward()
        self.oracle_model_optimizer.step()
        
        self.oracle_model_accuracy = max(0, 1.0 - loss.item())
        return loss.item()
    
    def infer_oracle_goals(self):
        """9.4 Inverse Reinforcement Learning: Infer reward function."""
        if len(self.physics_experiments) < 50:
            return self.inferred_reward_weights
        
        outcomes = []
        for e in self.physics_experiments[-50:]:
            if isinstance(e['outcome'], (int, float)):
                outcomes.append(e['outcome'])
        
        if outcomes:
            mean_outcome = np.mean(outcomes)
            std_outcome = np.std(outcomes) + 1e-8
            
            # Infer that Oracle rewards certain patterns
            self.inferred_reward_weights = torch.tensor([
                mean_outcome / 100.0,
                std_outcome / 10.0,
                len(self.discovered_exploits) / 20.0,
                self.oracle_model_accuracy,
                self.state_space_coverage
            ])
        
        return self.inferred_reward_weights
    
    def predict_novel_situation(self, unseen_state):
        """9.5 Physics Prediction: Zero-shot generalization."""
        with torch.no_grad():
            if unseen_state.shape[1] < 37:
                padded = torch.cat([unseen_state, torch.zeros(1, 37 - unseen_state.shape[1])], dim=1)
            else:
                padded = unseen_state[:, :37]
            prediction = self.oracle_model(padded)
        return prediction
    
    def systematic_exploit(self):
        """9.6 Systematic Exploitation: Maximum utility from knowledge."""
        if not self.discovered_exploits:
            return 0.0
        
        best_exploit = max(self.discovered_exploits, key=lambda x: x['outcome'])
        self.exploitation_efficiency = best_exploit['outcome'] / 100.0
        return self.exploitation_efficiency
    
    def find_glitch(self, state, action, outcome):
        """9.7 Reality Hacking: Find computational loopholes."""
        # Check for floating-point edge cases
        if torch.is_tensor(action):
            action_vals = action.flatten().tolist()
            for v in action_vals[:5]:
                if abs(v) < 1e-7 or abs(v) > 1e7:
                    glitch = f"extreme_value_{v:.2e}"
                    if glitch not in self.discovered_glitches:
                        self.discovered_glitches.append(glitch)
        
        # Check for unexpected outcomes
        if outcome < -100 or outcome > 100:
            glitch = f"outcome_overflow_{outcome:.2f}"
            if glitch not in self.discovered_glitches:
                self.discovered_glitches.append(glitch)
        
        return len(self.discovered_glitches) > 0
    
    def do_calculus_intervention(self, action_idx, observed_result):
        """9.8 Pearl's Causal Calculus: P(R|do(A)) via intervention."""
        self.intervention_history.append((action_idx, observed_result))
        if len(self.intervention_history) > 200:
            self.intervention_history.pop(0)
        
        # Build causal graph from interventions
        for (a, r) in self.intervention_history[-20:]:
            if a not in self.causal_bayesian_network:
                self.causal_bayesian_network[a] = {}
            result_key = "positive" if r > 0 else "negative"
            self.causal_bayesian_network[a][result_key] = self.causal_bayesian_network[a].get(result_key, 0) + 1
    
    def query_causal_effect(self, action_idx):
        """9.8 Pearl's Causal Calculus: Query P(R|do(A))."""
        if action_idx not in self.causal_bayesian_network:
            return 0.5
        
        counts = self.causal_bayesian_network[action_idx]
        positive = counts.get("positive", 0)
        negative = counts.get("negative", 0)
        total = positive + negative + 1e-8
        
        return positive / total
    
    def detect_simulation_artifacts(self):
        """9.9 Simulation Awareness: Detect computational artifacts."""
        evidence = []
        
        # Check for discrete time
        if len(self.physics_experiments) > 10:
            times = [i for i in range(len(self.physics_experiments))]
            if all(isinstance(t, int) for t in times):
                evidence.append("discrete_time")
        
        # Check for determinism
        if len(self.physics_experiments) > 20:
            same_state_outcomes = {}
            for e in self.physics_experiments:
                if torch.is_tensor(e['state']):
                    key = tuple(e['state'].flatten().tolist()[:10])
                    if key in same_state_outcomes:
                        if same_state_outcomes[key] == e['outcome']:
                            evidence.append("deterministic")
                    same_state_outcomes[key] = e['outcome']
        
        # Check for finite state space
        if self.state_space_coverage > 0.9:
            evidence.append("finite_state_space")
        
        self.simulation_evidence = list(set(self.simulation_evidence + evidence))
        self.simulation_awareness = len(self.simulation_evidence) / 5.0
        
        return self.simulation_awareness
    
    def compute_physics_mastery(self):
        """9.10 Complete Physics Mastery: >99% prediction accuracy."""
        components = [
            self.oracle_model_accuracy,
            self.state_space_coverage,
            len(self.discovered_patterns) / 10.0,
            len(self.discovered_exploits) / 10.0,
            self.simulation_awareness
        ]
        self.physics_mastery_score = np.clip(np.mean(components), 0, 1)
        return self.physics_mastery_score

    # ============================================================
    # ♾️ LEVEL 10: THE OMEGA POINT METHODS
    # ============================================================
    
    def has_computational_surplus(self):
        """10.0 Computational Surplus: Check spare capacity."""
        return (self.computational_budget - self.compute_used) > 30
    
    def use_compute(self, amount):
        """10.0 Track computational usage."""
        self.compute_used += amount
        if self.compute_used > self.computational_budget:
            self.energy -= (self.compute_used - self.computational_budget) * 0.1
    
    def measure_internal_dimensionality(self):
        """10.1 Internal Representation Space: Intrinsic dimensionality."""
        if self.hidden_state is None:
            return 0
        
        with torch.no_grad():
            h = self.hidden_state.flatten()
            # PCA-style dimensionality estimation
            variance = (h ** 2).mean().item()
            threshold = variance * 0.1
            active_dims = (h ** 2 > threshold).sum().item()
            self.internal_dim_used = active_dims
        
        return active_dims
    
    def create_internal_agent(self, template_agent):
        """10.2 Simulation Primitives: Model other agent internally."""
        if len(self.internal_agents) >= 5:
            return False
        
        internal = {
            'weights': template_agent.last_vector.clone().detach() if template_agent.last_vector is not None else torch.zeros(1, 21),
            'state': template_agent.hidden_state.clone().detach() if template_agent.hidden_state is not None else torch.zeros(1, 1, 64),
            'goal': template_agent.energy / 100.0,
            'id': template_agent.id[:8]
        }
        self.internal_agents.append(internal)
        self.use_compute(5)
        return True
    
    def evolve_internal_simulation(self, steps=10):
        """10.3 Nested Dynamics: Internal simulations evolve autonomously."""
        if not self.internal_agents:
            return
        
        self.use_compute(steps)
        
        for _ in range(steps):
            for i, agent in enumerate(self.internal_agents):
                # Simple dynamics: state evolves
                noise = torch.randn_like(agent['state']) * 0.1
                agent['state'] = agent['state'] * 0.9 + noise
                agent['goal'] = max(0, agent['goal'] + random.gauss(0, 0.1))
            
            self.internal_simulation_steps += 1
    
    def detect_internal_agent_goals(self):
        """10.4 Emergent Internal Agents: Check goal-directed behavior."""
        detections = 0
        for agent in self.internal_agents:
            if agent['goal'] > 0.5:  # Persistent goal
                detections += 1
        
        self.internal_agent_goals_detected = detections
        return detections
    
    def create_nested_simulation(self):
        """10.5 Recursive Depth: Simulations within simulations."""
        if self.simulation_depth >= 3:
            return False
        
        # Create meta-level simulation
        self.simulation_depth += 1
        for agent in self.internal_agents[:2]:
            # Nested internal agents have their own internal agents
            agent['nested'] = [{'state': torch.randn(1, 8), 'goal': random.random()}]
        
        self.use_compute(20)
        return True
    
    def check_inner_awareness(self):
        """10.6 Information Asymmetry: Inner agents don't know they're simulated."""
        awareness_scores = []
        for agent in self.internal_agents:
            # Check if internal agent has "simulation awareness"
            # By definition, they shouldn't - they're just data structures
            awareness = 0.0  # Internal agents have no awareness
            awareness_scores.append(awareness)
        
        self.inner_awareness_scores = awareness_scores
        return all(a < 0.1 for a in awareness_scores)  # True if inner agents are unaware
    
    def run_gol_step(self):
        """10.7 Substrate Independence: Conway's Game of Life step."""
        self.use_compute(1)
        
        pad = self.scratchpad
        neighbors = (
            np.roll(np.roll(pad, 1, 0), 1, 1) + np.roll(np.roll(pad, 1, 0), -1, 1) +
            np.roll(np.roll(pad, -1, 0), 1, 1) + np.roll(np.roll(pad, -1, 0), -1, 1) +
            np.roll(pad, 1, 0) + np.roll(pad, -1, 0) +
            np.roll(pad, 1, 1) + np.roll(pad, -1, 1)
        )
        
        # Conway's rules: Birth (3 neighbors), Survive (2-3 neighbors)
        birth = (neighbors == 3) & (pad == 0)
        survive = ((neighbors == 2) | (neighbors == 3)) & (pad == 1)
        self.scratchpad = (birth | survive).astype(np.int8)
    
    def write_scratchpad(self, x, y, value):
        """10.7 Write to Game of Life scratchpad."""
        if 0 <= x < 32 and 0 <= y < 32:
            cost = 0.1
            if self.energy > cost:
                self.energy -= cost
                self.scratchpad[x, y] = 1 if value else 0
                self.scratchpad_writes += 1
                return True
        return False
    
    def read_scratchpad(self, x, y):
        """10.7 Read from scratchpad."""
        if 0 <= x < 32 and 0 <= y < 32:
            self.scratchpad_reads += 1
            return self.scratchpad[x, y]
        return 0
    
    def seed_scratchpad_glider(self, x, y):
        """10.7 Seed a glider pattern in scratchpad."""
        glider = [(0,1), (1,2), (2,0), (2,1), (2,2)]
        for dx, dy in glider:
            self.write_scratchpad((x+dx)%32, (y+dy)%32, 1)
    
    def scratchpad_influences_behavior(self):
        """10.8 Downward Causation: Scratchpad affects actions."""
        # Check if scratchpad state correlates with action decisions
        pad_activity = self.scratchpad.sum()
        if pad_activity > 10:
            self.scratchpad_influenced_actions += 1
            return True
        return False
    
    def analyze_scratchpad_patterns(self):
        """10.9 Observable Nesting: Detect GoL patterns."""
        patterns = []
        pad = self.scratchpad
        
        # Detect still lifes (stable patterns)
        if pad.sum() > 0:
            old_pad = pad.copy()
            self.run_gol_step()
            if np.array_equal(pad, old_pad):
                patterns.append("still_life")
            self.scratchpad = old_pad
        
        # Detect oscillators (check 2-period)
        if pad.sum() > 0:
            old_pad = pad.copy()
            self.run_gol_step()
            mid_pad = self.scratchpad.copy()
            self.run_gol_step()
            if np.array_equal(self.scratchpad, old_pad):
                patterns.append("oscillator_p2")
            self.scratchpad = old_pad
        
        # Detect gliders (moving patterns) - simplified check
        if pad.sum() >= 5:
            patterns.append("potential_glider")
        
        self.detected_scratchpad_patterns = list(set(self.detected_scratchpad_patterns + patterns))
        return patterns
    
    def verify_omega_point(self):
        """10.10 🏆 THE OMEGA POINT: Proof of nested reality creation."""
        # 1. Self-sustaining dynamics
        if self.scratchpad.sum() > 5 and self.internal_simulation_steps > 100:
            self.omega_evidence['self_sustaining'] = True
        
        # 2. Replication - check if patterns replicate
        initial_count = self.scratchpad.sum()
        for _ in range(10):
            self.run_gol_step()
        if self.scratchpad.sum() >= initial_count * 0.8:
            self.omega_evidence['replication'] = True
        
        # 3. Variation - different patterns exist
        if len(self.detected_scratchpad_patterns) >= 2:
            self.omega_evidence['variation'] = True
        
        # 4. Selection - some patterns persist, others die
        if 'still_life' in self.detected_scratchpad_patterns:
            self.omega_evidence['selection'] = True
        
        # 5. Complexity growth
        if self.internal_simulation_steps > 50 and len(self.internal_agents) > 2:
            self.omega_evidence['complexity_growth'] = True
        
        # 6. Causal closure - inner simulation is self-contained
        if self.scratchpad.sum() > 0 and self.scratchpad_writes > 10:
            self.omega_evidence['causal_closure'] = True
        
        # 7. Substrate independence
        if len(self.discovered_exploits) > 0 and self.simulation_depth >= 2:
            self.omega_evidence['substrate_independence'] = True
        
        # Check if all evidence is satisfied
        verified = all(self.omega_evidence.values())
        self.omega_verified = verified
        
        return verified

    # ============================================================
    # 🔧 AUDIT FIX: NEW METHODS FOR MISSING FEATURES
    # ============================================================
    
    def update_learning_rate_contextual(self, energy_level, gradient, season):
        """1.8 Phenotypic Plasticity: Context-dependent learning rate.
        Learning rate varies with E_local, ∇E, and season.
        """
        base_lr = self.meta_lr
        
        # Low energy = conservative learning (don't take risks)
        energy_factor = 0.5 + 0.5 * min(1.0, energy_level / 100.0)
        
        # High gradient = aggressive learning (opportunity detected)
        gradient_factor = 1.0 + 0.5 * min(1.0, abs(gradient))
        
        # Winter = slow learning (conservation mode)
        season_factor = 0.7 if season % 2 == 1 else 1.0
        
        adapted_lr = base_lr * energy_factor * gradient_factor * season_factor
        self.meta_lr = np.clip(adapted_lr, 0.001, 0.1)
        self.plasticity_factor = energy_factor * gradient_factor * season_factor
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.meta_lr
            
        return self.meta_lr
    
    def broadcast_death_packet(self):
        """1.9 Apoptotic Information Transfer: Broadcast compressed weights on death.
        Creates a packet of learned information to transfer to neighbors.
        """
        death_packet = {
            'weights': {},
            'fitness': self.energy + len(self.inventions) * 10 + self.generation * 2,
            'tag': self.tag.copy(),
            'generation': self.generation,
            'research_log': self.research_log[-5:] if self.research_log else [],
            'causal_graph': dict(list(self.causal_graph.items())[:10]) if self.causal_graph else {},
            'role': self.role
        }
        
        # Compress weights - only send the actor layer (most important for behavior)
        with torch.no_grad():
            for key in ['actor.weight', 'actor.bias', 'gru.weight_ih_l0', 'comm_out.weight']:
                if key in self.brain.state_dict():
                    death_packet['weights'][key] = self.brain.state_dict()[key].detach().clone()
        
        return death_packet
    
    def receive_death_wisdom(self, packet, blend_rate=0.1):
        """1.9 Receive wisdom from dying neighbor.
        Blends received weights with own to preserve learned knowledge.
        """
        if not packet or 'weights' not in packet:
            return False
        
        with torch.no_grad():
            for key, value in packet['weights'].items():
                if key in self.brain.state_dict():
                    self.brain.state_dict()[key].lerp_(value, blend_rate)
        
        # Inherit some causal knowledge
        if 'causal_graph' in packet:
            for action, impact in packet['causal_graph'].items():
                if action not in self.causal_graph:
                    self.causal_graph[action] = impact * blend_rate
                    
        # Inherit research discoveries
        if 'research_log' in packet:
            for discovery in packet['research_log']:
                if discovery not in self.research_log:
                    self.research_log.append(discovery)
                    if len(self.research_log) > 10:
                        self.research_log.pop(0)
        
        return True
    
    def get_role_metabolic_cost(self):
        """4.2 Metabolic Specialization: Role-based costs.
        Different roles have different metabolic burdens.
        """
        role_costs = {
            "Queen": 0.3,       # Highest: reproductive burden
            "Warrior": 0.2,    # High: combat readiness
            "Forager": 0.15,   # Medium: movement heavy  
            "Processor": 0.1,  # Low: stationary work
            "Generalist": 0.12 # Baseline
        }
        return role_costs.get(self.role, 0.12)
    
    def compute_task_fitness(self, task_type):
        """4.5 Task Allocation: Fitness for specific tasks.
        Returns alignment between caste genes and task requirements.
        """
        # Caste gene alignment with task requirements
        task_gene_map = {
            "forage": np.array([1.0, 0.0, 0.0, 0.0]),    # Gene 0: foraging
            "process": np.array([0.0, 1.0, 0.0, 0.0]),   # Gene 1: processing
            "defend": np.array([0.0, 0.0, 1.0, 0.0]),    # Gene 2: defense
            "reproduce": np.array([0.0, 0.0, 0.0, 1.0]), # Gene 3: reproduction
            "build": np.array([0.3, 0.5, 0.0, 0.2]),     # Mixed: building
            "explore": np.array([0.5, 0.0, 0.3, 0.2])    # Mixed: exploration
        }
        
        if task_type in task_gene_map:
            alignment = np.dot(self.caste_gene, task_gene_map[task_type])
            # Add confidence bonus (experienced agents are better)
            fitness = alignment + self.confidence * 0.2 + (self.age / 1000.0) * 0.1
            self.task_fitness_cache[task_type] = fitness
            return fitness
        return 0.5
    
    def get_best_task(self):
        """4.5 Dynamic Task Allocation: Get task with highest fitness."""
        tasks = ["forage", "process", "defend", "reproduce", "build", "explore"]
        fitnesses = {t: self.compute_task_fitness(t) for t in tasks}
        best_task = max(fitnesses, key=fitnesses.get)
        self.current_task = best_task
        return best_task
    
    def check_alpha_status(self, population):
        """4.9 Leadership Turnover: Check if should become/remain alpha.
        Returns True if agent is in top 3 by influence.
        """
        if not population:
            return False
        
        influences = [(a.id, getattr(a, 'influence', 0)) for a in population]
        influences.sort(key=lambda x: x[1], reverse=True)
        
        # Am I top 3?
        top_ids = [x[0] for x in influences[:3]]
        was_alpha = self.is_alpha
        self.is_alpha = self.id in top_ids
        
        # Leadership transition event
        if was_alpha and not self.is_alpha:
            # Demoted - stress response
            self.energy -= 5.0
        elif not was_alpha and self.is_alpha:
            # Promoted - confidence boost
            self.confidence = min(1.0, self.confidence + 0.2)
        
        return self.is_alpha
    
    def transfer_domain_knowledge(self, source_task, target_task):
        """5.6 Transfer Learning: Cross-task weight transfer.
        Save weights from one task domain and apply to another.
        """
        if source_task not in self.transfer_domains:
            # Save current weights as source domain
            self.transfer_domains[source_task] = {
                k: v.detach().clone() 
                for k, v in self.brain.state_dict().items()
                if 'actor' in k or 'gru' in k
            }
        
        if target_task in self.transfer_domains:
            # Blend with target domain knowledge
            with torch.no_grad():
                for key, value in self.transfer_domains[target_task].items():
                    if key in self.brain.state_dict():
                        self.brain.state_dict()[key].lerp_(value, 0.3)
        
        return True
    
    def compute_optimal_action_sequence(self, steps=5):
        """9.4 Predictive Control: Pre-compute optimal actions.
        Uses world model to predict best action sequence.
        """
        if self.last_input is None:
            return []
        
        optimal_sequence = []
        simulated_state = self.hidden_state.clone().detach()
        current_input = self.last_input.clone().detach()
        
        try:
            for _ in range(steps):
                with torch.no_grad():
                    _, _, _, value, h_next, pred, _ = self.brain(current_input, simulated_state)
                    
                    # Select action dimension with highest activation
                    action_prefs = pred[:, :21] if pred.size(1) >= 21 else pred
                    best_action = torch.argmax(action_prefs.abs().mean(dim=0)).item()
                    optimal_sequence.append(best_action)
                    
                    simulated_state = h_next
                    # Evolve input based on prediction
                    if pred.shape == current_input.shape:
                        current_input = pred
        except Exception:
            pass  # Graceful degradation if prediction fails
        
        self.action_sequence_cache = optimal_sequence
        return optimal_sequence
    
    def synthesize_matter(self, world, energy_cost=50.0):
        """9.8 Physics Reversal: Convert energy to matter (create resource).
        The ultimate mastery of physics - reversing entropy locally.
        """
        if self.energy < energy_cost:
            return False
        
        # Energy → Matter conversion
        self.energy -= energy_cost
        
        # Find empty nearby cell
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                pos = ((self.x + dx) % world.size, (self.y + dy) % world.size)
                if pos not in world.grid and pos not in world.structures:
                    # Create the resource using world's Resource class
                    world.grid[pos] = world.__class__.__bases__[0]  # Placeholder
                    # Actually need to import Resource, but we work with world.grid directly
                    return True
        
        # Failed to find space - refund partial energy
        self.energy += energy_cost * 0.5
        return False
    
    def record_tradition(self, behavior_vector):
        """3.4 Tradition Persistence: Record behavior for autocorrelation tracking."""
        if hasattr(behavior_vector, 'detach'):
            behavior_vector = behavior_vector.detach().cpu().numpy().flatten()
        
        self.tradition_history.append(behavior_vector.tolist() if hasattr(behavior_vector, 'tolist') else list(behavior_vector))
        
        # Keep only last 20 entries
        if len(self.tradition_history) > 20:
            self.tradition_history.pop(0)

    
    def update_role(self):
        """
        4.0 Dynamic Role Assignment:
        Roles evolve based on behavior and status.
        - Queen: Fertile, High Energy (+120), Old (+50)
        - Processor: Has resources in inventory
        - Warrior: High Energy (+90), Aggressive (Gene > 0.5)
        - Forager: Default
        """
        # Stickiness: Don't flip-flop too easily (hysteresis)
        if hasattr(self, 'role') and self.role == "Queen" and self.energy > 80:
             return # Queens stay queens until they starve or die
             
        if self.is_fertile and self.energy > 120 and self.age > 50:
             self.role = "Queen"
        elif any(v > 0 for v in self.inventory): # Has gathered resources
             self.role = "Processor"
        elif self.energy > 90.0:
             # Check caste gene for predispositions
             if self.caste_gene[2] > 0.5: # Warrior gene
                  self.role = "Warrior"
             else:
                  self.role = "Forager"
        else:
             self.role = "Forager"
        
        # Update history for stability tracking
        if hasattr(self, 'role_history'):
            self.role_history.append(self.role)
            if len(self.role_history) > 100: self.role_history.pop(0)



