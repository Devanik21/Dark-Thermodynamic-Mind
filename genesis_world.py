import numpy as np
import torch
import torch.nn as nn
import random
import math
import streamlit as st

# ============================================================
# 🌌 DYNAMIC PHYSICS CONSTANTS
# ============================================================
GRID_SIZE = 40
SIGNAL_DIM = 16
MAX_ENERGY = 100000.0 # Effectively Infinite (Type II Civilization Potential)
# "Easy mode" - metabolic cost is low, but stupidity kills
METABOLIC_COST = 0.08 
SEASON_LENGTH = 20 # Shortened winter (Nobel Optimization)

# ============================================================
# 🔮 THE PHYSICS ORACLE (The Laws of Nature)
# ============================================================
class PhysicsOracle(nn.Module):
    """
    The Black Box of Reality.
    Maps User Will (21D) -> Physical Effect.
    Biased: 61% Positive, 39% Negative.
    """
    def __init__(self):
        super().__init__()
        # Input: 21 (Will) + 16 (Local Matter Signal) = 37 Dimensions
        self.layers = nn.Sequential(
            nn.Linear(37, 64),
            nn.Tanh(), # Non-linear chaotic mixing
            nn.Linear(64, 64),
            nn.SiLU(), # Complex activation
            nn.Linear(64, 5) # Output Effects
        )
        
        # Balanced Initialization for maximum Chaos
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Bias the "Energy" output (Index 0) SLIGHTLY positive
        # Was 0.5 (Too safe). Now 0.1 (Survival requires finding the peaks)
        with torch.no_grad():
            self.layers[-1].bias[0] = 0.0 
            self.layers[-1].bias[4] = -0.3 # Harsher drain on interaction
            
    def forward(self, vector_21, matter_signal_16):
        x = torch.cat([vector_21, matter_signal_16], dim=1)
        return self.layers(x)

# ============================================================
# ⚛️ ENTITIES
# ============================================================
class Entity:
    def __init__(self, x, y, entity_type):
        self.x = x
        self.y = y
        self.type = entity_type
        self.exists = True

class Resource(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 'resource')
        # Level 2.8: Trade Emergence (Resource Types)
        # 0: Red (Standard), 1: Green (Rich), 2: Blue (Rare/Catalyst)
        self.type = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        # Signal Spectrum based on type
        self.signal = torch.zeros(SIGNAL_DIM)
        start_ch = self.type * 4
        self.signal[start_ch:start_ch+4] = torch.rand(4) * 0.8 + 0.2
        self.signal = torch.nn.functional.normalize(self.signal, dim=0)

    def get_nutrition(self, current_season):
        # Summer (Even) favors Red/Green
        # Winter (Odd) favors Blue
        base = 30.0
        
        is_summer = (current_season % 2 == 0)
        
        if is_summer:
            if self.type == 0 or self.type == 1: return base
            # Blue in Summer: Low value, not toxic
            return 10.0 
        else: # Winter
            if self.type == 2: return base * 8.0 # Blue is winter-survival fuel
            
            # Nuanced Winter Scarcity:
            # "Elite Mode": Red/Green are 25-30 vs 30. Barely different.
            # 0.001% chance of toxicity (-1)
            if random.random() < 0.00001:
                return -0.2
            return random.uniform(25.0, 35.0)

class MegaResource(Entity):
    """4.8 Distributed Cognition: Requires multiple agents or high synergy."""
    def __init__(self, x, y):
        super().__init__(x, y, 'mega_resource')
        self.strength = 100.0
        self.signal = torch.ones(SIGNAL_DIM) * 0.5
    
    def get_nutrition(self, current_season):
        return 150.0 # High payout

# ============================================================
# 🏗️ LEVEL 6: GEO-ENGINEERING STRUCTURES
# ============================================================
class Structure(Entity):
    """6.2 Base class for agent-built persistent structures."""
    def __init__(self, x, y, structure_type, builder_id):
        super().__init__(x, y, structure_type)
        self.structure_type = structure_type
        self.builder_id = builder_id
        self.age = 0
        self.created_tick = 0
        self.durability = 100.0
        self.signal = torch.zeros(SIGNAL_DIM)
        self.signal[12:16] = 0.8  # Structures emit unique spectrum
    
    def decay(self, rate=0.1):
        """Structures decay over time."""
        self.age += 1
        self.durability -= rate
        if self.durability <= 0:
            self.exists = False
        return self.exists

class Trap(Structure):
    """6.3 Trap Construction: Captures energy from passing agents."""
    def __init__(self, x, y, builder_id, harvest_rate=0.2):
        super().__init__(x, y, "trap", builder_id)
        self.harvest_rate = min(0.5, harvest_rate)
        self.stored_energy = 0.0
        self.victims = []
    
    def harvest(self, agent):
        """Harvest energy from agent that steps on trap."""
        if agent.id == self.builder_id:
            return 0  # Don't trap builder
        energy_taken = agent.energy * self.harvest_rate
        if hasattr(agent, 'shield_strength'):
            energy_taken *= (1 - agent.shield_strength)
        agent.energy -= energy_taken
        self.stored_energy += energy_taken
        self.victims.append(agent.id)
        return energy_taken
    
    def collect(self, agent):
        """Builder collects stored energy."""
        if agent.id == self.builder_id:
            collected = self.stored_energy * 0.9
            self.stored_energy = 0.0
            agent.energy += collected
            return collected
        return 0.0

class Barrier(Structure):
    """6.4 Defensive Architecture: Blocks or filters movement."""
    def __init__(self, x, y, builder_id, filter_mode="all"):
        super().__init__(x, y, "barrier", builder_id)
        self.filter_mode = filter_mode  # "all", "enemies", "low_energy"
        self.blocked_count = 0
    
    def allows_passage(self, agent):
        """Check if agent can pass through."""
        if agent.id == self.builder_id:
            return True
        if self.filter_mode == "all":
            self.blocked_count += 1
            return False
        if self.filter_mode == "enemies":
            # Allow same-generation agents through
            return hasattr(agent, 'generation') and agent.generation > 0
        if self.filter_mode == "low_energy":
            return agent.energy > 50
        return True

class Battery(Structure):
    """6.8 Energy Storage: Environmental energy caching."""
    def __init__(self, x, y, builder_id, capacity=500.0):
        super().__init__(x, y, "battery", builder_id)
        self.capacity = capacity
        self.stored_energy = 0.0
        self.authorized_users = {builder_id}
    
    def deposit(self, agent, amount):
        """Store energy in battery."""
        if self.stored_energy + amount > self.capacity:
            actual = self.capacity - self.stored_energy
        else:
            actual = amount
        if agent.energy >= actual:
            agent.energy -= actual
            self.stored_energy += actual * 0.9  # 10% loss
            return actual
        return 0.0
    
    def withdraw(self, agent):
        """Retrieve all stored energy."""
        if agent.id in self.authorized_users:
            amount = self.stored_energy * 0.9  # 10% loss
            self.stored_energy = 0.0
            agent.energy += amount
            return amount
        return 0.0
    
    def authorize(self, agent_id):
        """Add agent to authorized users."""
        self.authorized_users.add(agent_id)

class Cultivator(Structure):
    """6.5 Resource Cultivation: Biases resource spawn probability nearby."""
    def __init__(self, x, y, builder_id, boost_radius=2, boost_strength=0.3):
        super().__init__(x, y, "cultivator", builder_id)
        self.boost_radius = boost_radius
        self.boost_strength = boost_strength
        self.spawned_count = 0
    
    def get_influenced_tiles(self, world_size):
        """Return list of tiles influenced by this cultivator."""
        tiles = []
        for dx in range(-self.boost_radius, self.boost_radius + 1):
            for dy in range(-self.boost_radius, self.boost_radius + 1):
                nx = (self.x + dx) % world_size
                ny = (self.y + dy) % world_size
                tiles.append((nx, ny))
        return tiles

class InfrastructureNetwork:
    """6.9 Infrastructure Networks: Connected structure systems."""
    def __init__(self, network_id):
        self.id = network_id
        self.structures = set()  # Structure positions
        self.members = set()  # Agent IDs
        self.efficiency_bonus = 0.0
    
    def add_structure(self, x, y):
        """Add structure to network."""
        self.structures.add((x, y))
        self._update_efficiency()
    
    def add_member(self, agent_id):
        """Add agent to network."""
        self.members.add(agent_id)
    
    def _update_efficiency(self):
        """Update network efficiency based on connectivity."""
        if len(self.structures) < 2:
            self.efficiency_bonus = 0.0
            return
        # Efficiency grows with network size (diminishing returns)
        self.efficiency_bonus = min(0.5, 0.1 * np.log(len(self.structures) + 1))
    
    def get_transfer_rate(self):
        """Energy can be transferred through network with efficiency bonus."""
        return 0.8 + self.efficiency_bonus

class TerrainModification:
    """6.7 Terraforming: Permanent terrain changes."""
    def __init__(self, x, y, terrain_type, modifier_id):
        self.x = x
        self.y = y
        self.terrain_type = terrain_type  # "fertile", "barren", "water", "elevated"
        self.modifier_id = modifier_id
        self.age = 0

    def update_age(self):
        self.age += 1
    
    def get_spawn_modifier(self):
        """Return spawn rate modifier for this terrain."""
        modifiers = {
            "fertile": 2.0,
            "barren": 0.1,
            "water": 0.0,
            "elevated": 0.5
        }
        return modifiers.get(self.terrain_type, 1.0)
    
    def get_movement_cost(self):
        """Return movement cost for this terrain."""
        costs = {
            "fertile": 0.5,
            "barren": 1.5,
            "water": 3.0,
            "elevated": 2.0
        }
        return costs.get(self.terrain_type, 1.0)

# ============================================================
# 🌍 THE QUANTUM WORLD
# ============================================================
# ============================================================
# 🌍 THE QUANTUM WORLD
# ============================================================
class GenesisWorld:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = {} 
        self.agents = {} 
        self.time_step = 0
        self.current_season = 0
        self.season_timer = 0
        
        # 🌐 PHASE 13: "TURING" UPGRADE (16D Pheromone Grid)
        self.pheromone_grid = np.zeros((size, size, SIGNAL_DIM))
        
        # 3.3 Environmental Marking (Stigmergy)
        # Channels: 0:Danger(Red), 1:Resource(Green), 2:Sacred(Blue)
        self.meme_grid = np.zeros((size, size, 3)) 
        
        # 🔗 PHASE 15: "SYMBIOGENESIS" UPGRADE (Elastic Bonds)
        self.bonds = set() # Set of tuples (frozenset of agent IDs)
        
        # 1.4 Scarcity Scaling
        self.scarcity_lambda = 0.01
        self.discovered_physics_exploits = [] # 9.2 Collective Exploit Tracking
        self.base_spawn_rate = 15 # GOLDEN ERA: Increased from 10 to 15
        
        # 1.10 Entropy Tracking
        self.system_entropy = 0.0
        self.agent_entropy = 0.0
        self.dissipated_energy = 0.0
        self.oracle = PhysicsOracle()
        # Freeze the laws (God does not play dice twice)
        for p in self.oracle.parameters():
            p.requires_grad = False
            
        # 5.6 Collective Optimization
        self.collective_values = {"Efficiency": 0.5, "Growth": 0.5}
        self.fitness_landscape_shift = 0.0

        # ============================================================
        # 🌍 LEVEL 6: GEO-ENGINEERING STATE
        # ============================================================
        self.structures = {}  # {(x,y): Structure object}
        self.networks = {}  # {network_id: InfrastructureNetwork}
        self.terrain_modifications = {}  # {(x,y): TerrainModification}
        self.cultivator_map = {}  # {(x,y): spawn_boost}
        self.weather_amplitude = 1.0  # Collective weather control modifier
        
        # ============================================================
        # 🐝 LEVEL 7: COLLECTIVE MANIFOLD STATE
        # ============================================================
        self.kuramoto_order_parameter = 0.0  # |<e^{iθ}>| sync measure
        self.collective_hidden_state = torch.zeros(1, 1, 64)  # Blended population state
        self.gradient_pool = []  # Shared gradient pool for federated learning
        self.cognitive_modules = {  # Specialist agents by role
            "vision": [],
            "planning": [],
            "memory": [],
            "motor": []
        }
        self.consensus_registry = {}  # {proposal_id: {votes, result}}
        self.protocol_convergence = 0.0  # How aligned communication protocols are
        self.hive_phi = 0.0  # Collective Φ (sum of agent contributions)
        
        # ============================================================
        # 💭 LEVEL 8: ABSTRACT REPRESENTATION STATE
        # ============================================================
        self.population_phi = 0.0  # Average Φ across population
        self.consciousness_count = 0  # Number of verified conscious agents
        self.strange_loop_count = 0  # Agents with active strange loops
        self.qualia_registry = {}  # {experience_type: activation_prototype}
        
        # ============================================================
        # ⚛️ LEVEL 9: UNIVERSAL HARMONIC RESONANCE STATE
        # ============================================================
        self.collective_oracle_model_accuracy = 0.0  # Best agent's Oracle approximation
        self.discovered_physics_patterns = []  # Patterns found by any agent
        self.collective_simulation_awareness = 0.0  # Max agent awareness
        self.causal_graph_collective = {}  # Merged causal graphs
        
        # ============================================================
        # ♾️ LEVEL 10: THE OMEGA POINT STATE
        # ============================================================
        self.omega_candidates = []  # Agent IDs that may have achieved Omega
        self.omega_achieved = False  # Global verification flag
        self.global_scratchpad_activity = 0  # Total scratchpad writes
        self.nested_simulation_depth_max = 0  # Maximum nesting observed
        
        # ============================================================
        # 🔧 AUDIT FIX: NEW STATE VARIABLES
        # ============================================================
        # 1.10 Enhanced Entropy Verification
        self.entropy_verification_count = 0
        
        # 3.4 Tradition Persistence Tracking
        self.tradition_tracker = {}  # {generation: [behavior_vectors]}
        self.tradition_persistence_verified = False
        
        # 3.5 Cultural Drift
        self.cultural_divergence = 0.0  # KL divergence between spatial groups
        
        # 3.8 Cultural Ratchet
        self.invention_history = []  # [{type: 'discovery'/'loss', tick, count}]
        self.cultural_ratchet_verified = False
        
        # 4.9 Leadership Tracking
        self.current_alphas = []  # List of alpha agent IDs
        
        # 6.9 Planetary Engineering
        self.planetary_structure_coverage = 0.0
        self.planetary_engineering_verified = False
        
        # 6.10 Type II Civilization
        self.structure_energy_ratio = 0.0
        self.type_ii_verified = False
        
        # 8.0 Symbol Grounding
        self.symbol_grounding_r2 = 0.0
        self.symbol_grounding_verified = False

    def collective_memory_retrieval(self):
        """4.9 Collective Memory: Agents with low confidence query the hive."""
        # Only run occasionally to save compute
        if self.time_step % 10 != 0: return

        for agent in self.agents.values():
            if agent.confidence < 0.3: # Confused/Forgetting
                 # Find a confident neighbor (Mentor)
                 best_mentor = None
                 max_conf = 0.5
                 
                 neighbors = [
                     self.agents[oid] for oid in self.agents 
                     if oid != agent.id and abs(self.agents[oid].x - agent.x) <= 2 and abs(self.agents[oid].y - agent.y) <= 2
                 ]
                 
                 for n in neighbors:
                     if n.confidence > max_conf:
                         max_conf = n.confidence
                         best_mentor = n
                 
                 if best_mentor:
                     # Restore deleted/decayed weights from mentor
                     if hasattr(agent, 'restorative_imitation'):
                        agent.restorative_imitation(best_mentor)
            
            
    def spawn_resource(self):
        x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
        if (x, y) not in self.grid:
            self.grid[(x, y)] = Resource(x, y)

    def get_local_signal(self, x, y):
        if (x, y) in self.grid:
            return self.grid[(x, y)].signal
        return torch.zeros(SIGNAL_DIM)

    def get_pheromone(self, x, y):
        # Read the chemical signal at this location (16D Vector)
        return torch.tensor(self.pheromone_grid[x, y], dtype=torch.float32)

    def get_energy_gradient(self, x, y):
        """1.7 Stress Response: Detect energy flux gradient."""
        # Check neighbors
        gradients = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = (x + dx) % self.size, (y + dy) % self.size
                if (nx, ny) in self.grid:
                    # Positive for food, negative for poison (based on current season)
                    gradients.append(self.grid[(nx, ny)].get_nutrition(self.current_season))
                else:
                    gradients.append(0.0)
        return torch.tensor([np.mean(gradients)], dtype=torch.float32)

    def update_pheromones(self):
        """
        Simulates diffusion and evaporation of 16D chemical signals.
        """
        grid = self.pheromone_grid
        
        # Vectorized diffusion using numpy rolling for all channels
        up = np.roll(grid, 1, axis=0)
        down = np.roll(grid, -1, axis=0)
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)
        
        diffused = (grid + up + down + left + right) / 5.0
        
        # Evaporation (Decay)
        self.pheromone_grid = diffused * 0.95 

    def metabolic_osmosis(self):
        """
        Level 2.5 Resource Sharing & Level 2.7 Punishment.
        Agents sharing a bond transfer energy to equalize gradients, modulated by Trust.
        """
        active_bonds = list(self.bonds)
        
        for bond in active_bonds:
            id_a, id_b = list(bond)
            if id_a not in self.agents or id_b not in self.agents:
                self.bonds.remove(bond)
                continue
                
            a, b = self.agents[id_a], self.agents[id_b]
            
            # 1. Kinship & Trust
            kinship = 1.0 - np.linalg.norm(a.tag - b.tag)
            kinship = np.clip(kinship, 0, 1)
            
            # Flow High -> Low
            if a.energy > b.energy: donor, receiver = a, b
            else: donor, receiver = b, a
            
            delta = (donor.energy - receiver.energy) * 0.05 # 5% per tick
            efficiency = 0.5 + (0.5 * kinship) # 100% efficient if same tribe
            
            if donor.energy > delta:
                donor.energy -= delta
                receiver.energy += delta * efficiency
                self.dissipated_energy += delta * (1.0 - efficiency)
                
                # Trust Gain
                receiver.social_memory[donor.id] = receiver.social_memory.get(donor.id, 0) + 0.1

    def resolve_quantum_state(self, agent, reality_vector, emit_vector=None, adhesion=0.0, punish=0.0, trade=0.0):
        """
        The Agent casts a spell (Vector). The Oracle decides what happens.
        Includes Level 2 Social Logic.
        """
        # 1. 2.3 Costly Signaling
        if emit_vector is not None:
             # Signaling costs energy proportional to signal complexity (variance)
             signal_complexity = float(torch.var(emit_vector).item())
             signal_cost = signal_complexity * 2.0
             
             # 2.3 Zahavi Strict Check: Verify Proof of Work
             # If complexity > threshold, we check the nonce.
             if signal_complexity > 0.05:
                 start_nonce = getattr(agent, 'last_nonce', 0)
                 import hashlib
                 # Quantize for consistency matches Agent
                 vec_bytes = (emit_vector * 100).long().cpu().numpy().tobytes()
                 candidate = f"{start_nonce}".encode() + vec_bytes
                 h = hashlib.sha256(candidate).hexdigest()
                 
                 # Verify Difficulty 1 ("0")
                 if not h.startswith("0"):
                     # Fake signal! Determine it's cheap talk.
                     # Dampen the signal so it doesn't propagate as effectively
                     emit_vector = emit_vector * 0.1
                     # But still charge energy (cheating isn't free)
                     outcome_log = "👺 FAKE SIGNAL DETECTED"
                 else:
                     # Honest signal. Add extra metabolic cost for the computation
                     agent.energy -= 0.5 
             
             agent.energy -= signal_cost
             
             # 3.8 Symbolic Reference (Displacement)
             # If Channel 15 > 0.5, project signal to offset coordinates (Ch 13, 14)
             target_x, target_y = agent.x, agent.y
             displacement_flag = emit_vector[15].item()
             if displacement_flag > 0.5:
                 # Map 0-1 to -5 to +5 offset
                 dx = int((emit_vector[13].item() - 0.5) * 10)
                 dy = int((emit_vector[14].item() - 0.5) * 10)
                 target_x = (agent.x + dx) % self.size
                 target_y = (agent.y + dy) % self.size
                 # Higher cost for projection
                 agent.energy -= 0.5
             
             self.pheromone_grid[target_x, target_y] += emit_vector.detach().numpy()
             np.clip(self.pheromone_grid[target_x, target_y], 0, 1.0, out=self.pheromone_grid[target_x, target_y])

        # 2. 2.4 Coalition & 2.5 Resource Sharing: BOND LOGIC
        if adhesion > 0.5:
            for other_id, other in self.agents.items():
                if other_id != agent.id:
                    dist = math.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                    if dist < 1.5:
                        # Bonding with different tribes is costly (2.4)
                        tag_dist = np.linalg.norm(agent.tag - other.tag)
                        bond_cost = 0.5 + tag_dist * 2.0
                        if agent.energy > bond_cost:
                            agent.energy -= bond_cost
                            self.bonds.add(frozenset([agent.id, other_id]))
        elif adhesion < 0.2:
            to_remove = [b for b in self.bonds if agent.id in b]
            for b in to_remove:
                self.bonds.remove(b)
                agent.energy -= 1.0 

        # 3. 2.7 Punishment
        outcome_log = "✨ IDLE"
        if punish > 0.7:
             # Find a neighbor to punish
             for other_id, other in self.agents.items():
                if other_id != agent.id:
                    dist = math.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                    if dist < 1.5:
                        cost = 5.0
                        damage = 15.0
                        if agent.energy > cost:
                            agent.energy -= cost
                            other.energy -= damage
                            other.social_memory[agent.id] = other.social_memory.get(agent.id, 0) - 1.0 # Trust loss
                            outcome_log = f"⚔️ PUNISHED {other_id[:4]}"
                            break

        # 4. 2.8 Trade
        if trade > 0.7:
            # Look for trade partners
            for b in self.bonds:
                if agent.id in b:
                    other_id = list(b - {agent.id})[0]
                    if other_id in self.agents:
                        partner = self.agents[other_id]
                        # Swap tokens if they have different types
                        for i in range(3):
                            for j in range(3):
                                if i != j and agent.inventory[i] > 0 and partner.inventory[j] > 0:
                                    agent.inventory[i] -= 1
                                    agent.inventory[j] += 1
                                    partner.inventory[j] -= 1
                                    partner.inventory[i] += 1
                                    agent.social_memory[other_id] = agent.social_memory.get(other_id, 0) + 0.5 # Trust gain
                                    outcome_log = f"🤝 TRADED with {other_id[:4]}"
                                    break

        # 5. Query Oracle
        loc = (agent.x, agent.y)
        local_sig = self.get_local_signal(*loc).unsqueeze(0)
        
        # --- PHASE 16: LEVEL 4 TENSOR FUSION & ROLES ---
        final_reality_vector = reality_vector
        if agent.is_fused and agent.fused_partner and agent.fused_partner.id in self.agents:
            partner = agent.fused_partner
            dist_p = math.sqrt((agent.x - partner.x)**2 + (agent.y - partner.y)**2)
            if dist_p < 2.0:
                # 4.7 Fusion: Additive reality vectors (Conceptual merge)
                final_reality_vector = reality_vector + partner.last_vector
        
        with torch.no_grad():
            effects = self.oracle(final_reality_vector, local_sig)[0] 
        
        # 6. Decode Effects
        energy_flux = effects[0].item() * 15.0 
        dx_raw = effects[1].item()
        dy_raw = effects[2].item()
        transmute = effects[3].item()
        
        # --- A. MOVEMENT ---
        if abs(energy_flux) < 2.0: 
            dx = 1 if dx_raw > 0.5 else (-1 if dx_raw < -0.5 else 0)
            dy = 1 if dy_raw > 0.5 else (-1 if dy_raw < -0.5 else 0)
            if dx != 0 or dy != 0:
                agent.x, agent.y = (agent.x + dx) % self.size, (agent.y + dy) % self.size
                agent.energy -= 0.1 
                outcome_log = "MOVE"
                
        # --- B. ENERGY & RECOURSES ---
        else:
            # 4.3 Supply Chain & Role Bonuses
            bonus = 1.0
            if agent.role == "Forager": bonus = 1.2
            elif agent.role == "Processor": bonus = 1.5 if any(v > 0 for v in agent.inventory) else 0.8
            
            final_flux = energy_flux * bonus
            agent.energy += final_flux
            
            if final_flux > 0:
                outcome_log = f"⚡ {agent.role} FLUX (+)"
                if loc in self.grid:
                    res = self.grid[loc]
                    # 4.8 Distributed Cognition Check
                    if isinstance(res, MegaResource):
                        if final_reality_vector.sum() < 2.0:
                            agent.energy -= 10.0 # Failed to harvest
                            outcome_log = "❌ TOO WEAK FOR MEGA"
                            return final_flux, outcome_log
                    
                    # 2.8 Token Collection (Standard Resource)
                    try:
                        idx = int(res.type)
                        agent.inventory[idx] += 1
                        # Synergy Bonus: Complete set (R,G,B) gives +30 Energy
                        if all(count > 0 for count in agent.inventory):
                            agent.energy += 30.0
                            for i in range(3): agent.inventory[i] -= 1
                            outcome_log = "🌟 SYNERGY BONUS!"
                        else:
                            outcome_log = f"😋 CONSUMED {['Red','Green','Blue'][idx]}"
                    except (ValueError, TypeError):
                        # This handles 'mega_resource' or any other non-standard entity
                        agent.energy += 150.0 
                        outcome_log = "💎 MEGA-RESOURCE HARVESTED!"
                    
                    del self.grid[loc]
            else: outcome_log = "🔥 NEGATIVE FLUX (-)"
        
        return energy_flux, outcome_log

    def step(self):
        self.time_step += 1
        self.season_timer += 1
        
        # 1.4 Scarcity: Exponential decay of spawn rate
        # Nobel-Level Fix: floor at 0.7 (Always 70% abundance) to prevent extinction
        current_spawn_prob = max(0.7, np.exp(-self.scarcity_lambda * self.time_step))
        
        # 1.6 Circadian Rhythms: Environment Phase
        self.env_phase = (self.time_step / SEASON_LENGTH) * 2 * np.pi
        
        # Phase 13: Biology Update
        self.update_pheromones()
        
        # Nobel Safeguard: Universal Fertility (Phenotypic Plasticity)
        # If population crashes, everyone becomes a Queen to save the species.
        n_pop = len(self.agents)
        if n_pop < 50:
            for agent in self.agents.values():
                agent.is_fertile = True
        
        # Nobel Adaptive Spawning: Smooth Continuous Scaling
        # Multiplier = 1 + 10 * exp(-pop/100)
        # Pop 10 -> 10x
        # Pop 100 -> 4.6x
        # Pop 300 -> 1.5x
        # Pop 500 -> 1.0x
        adaptive_rate = self.base_spawn_rate * (1.0 + 10.0 * np.exp(-n_pop / 100.0))
        
        for _ in range(int(adaptive_rate * current_spawn_prob)):
            self.spawn_resource()

        # Phase 15: Symbiosis Update
        self.metabolic_osmosis()
        
        # 4.9 Collective Memory: Retrieve lost knowledge
        self.collective_memory_retrieval()
        
        self._update_entropy_metrics()
        
        # 5.6 Collective Optimization
        if self.time_step % 100 == 0:
            self._update_fitness_landscape()
            
        # --- 3.2 HORIZONTAL NEURAL TRANSFER (Viral Propagation) ---
        # Highly fit agents (Energy > 90) spontaneously create "Weight Viruses"
        # and sneeze them onto their neighbors.
        if self.time_step % 50 == 0:
            for agent in list(self.agents.values()):
                if agent.energy > 90.0:
                    # Create virus packet
                    packet = agent.create_weight_packet()
                    # Broadcast to neighbors
                    neighbors = [
                        a for a in self.agents.values() 
                        if a.id != agent.id and abs(a.x - agent.x) <= 2 and abs(a.y - agent.y) <= 2
                    ]
                    for n in neighbors:
                         n.receive_infection(packet)
                         # Log the infection event sparingly
                         if random.random() < 0.05:
                            st.session_state.event_log.insert(0, {
                                "Tick": self.time_step,
                                "Agent": n.id,
                                "Event": f"🦠 INFECTED by {agent.id[:4]}",
                                "Vector": [0]*21
                            })
        
        if self.season_timer >= SEASON_LENGTH:
            self.current_season += 1
            self.season_timer = 0
            # 1.4 Scarcity applied to seasonal spawn
            for _ in range(int(20 * current_spawn_prob)): 
                self.spawn_resource()
        
        if self.time_step % 2 == 0:
            for _ in range(5): self.spawn_resource()
        
        # 4.8 Spawn MegaResource rarely
        if self.time_step % 100 == 0:
            mx, my = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.grid[(mx, my)] = MegaResource(mx, my)

    def _update_entropy_metrics(self):
        """1.10 Complete Entropy Defiance: Track system-wide entropy changes."""
        # 1. Agent Weight Entropy (Neural Compression)
        if self.agents:
            weights = []
            for a in list(self.agents.values()):
                weights.append(a.calculate_weight_entropy())
            self.agent_entropy = np.mean(weights)
        
        # 2. Environmental Entropy (Resource Distribution)
        # Using a simple grid occupancy entropy
        occ = np.zeros(self.size * self.size)
        for loc in self.grid:
            idx = loc[0] * self.size + loc[1]
            occ[idx] = 1
        p = (occ.sum() + 1e-8) / len(occ)
        env_entropy = -(p * np.log2(p + 1e-8) + (1-p) * np.log2(1-p + 1e-8))
        
        # 3. 1.10 Complete Entropy Defiance Verification
        # Total S = S_agents + S_environment
        # We need dS_agents < 0 and dS_total > 0 (Global dissipation)
        # Since I am not tracking dS_dissipated strictly yet, I use dissipated_energy as proxy for S_produced.
        
        current_agent_S = self.agent_entropy
        current_env_S = env_entropy
        
        # Store previous values for derivative
        if not hasattr(self, 'last_agent_S'):
            self.last_agent_S = current_agent_S
            self.last_system_S = self.system_entropy
            self.entropy_verified = False
        else:
            dS_agents = current_agent_S - self.last_agent_S
            dS_env = current_env_S - self.system_entropy # Actually system_S IS env_S here
            
            # S_dissipated is monotonically increasing, so dS_dissipated/dt > 0 always.
            # Verification Condition:
            # Agent Entropy is decreasing (ordering themselves)
            # while Universe Entropy (Env + Heat) is increasing.
            
            # Heat produced this step ~ dissipated_energy delta
            # We track cumulative, so we need delta
            if not hasattr(self, 'last_dissipated'): self.last_dissipated = 0.0
            d_heat = self.dissipated_energy - self.last_dissipated
            self.last_dissipated = self.dissipated_energy
            
            # Strict Check
            # We smooth it over time to avoid noise failures
            if dS_agents < 0 and (dS_env + d_heat) > 0:
                 self.entropy_verified = True
            
            self.last_agent_S = current_agent_S
        
        self.system_entropy = env_entropy

    def _update_fitness_landscape(self):
        """5.6 Collective Optimization: Population evolves the fitness function."""
        # Agents "vote" via their behavior.
        # High Trade = Vote for Efficiency.
        # High Movement = Vote for Growth/Exploration.
        
        if not self.agents: return
        
        avg_trade = np.mean([a.last_value.item() for a in self.agents.values() if a.last_value is not None])
        # Use simple heuristics
        
        eff_vote = 0.0
        growth_vote = 0.0
        
        for a in self.agents.values():
            if hasattr(a, 'role'):
                if a.role == "Processor": eff_vote += 1.0
                if a.role == "Forager": growth_vote += 1.0
                
        total = eff_vote + growth_vote + 1e-8
        self.collective_values["Efficiency"] = eff_vote / total
        self.collective_values["Growth"] = growth_vote / total
        
        # Shift Oracle Logic
        # If Efficiency is valued, Processors get bonus flux.
        # If Growth is valued, Foragers get bonus flux.
        # This is strictly "Collective Optimization" - the rules change to fit the dominant strategy.
        
        # We model this by shifting the 'bias' of the Oracle's last layer
        # Index 0 is Energy.
        bias_shift = (self.collective_values["Efficiency"] - 0.5) * 0.1
        self.fitness_landscape_shift = bias_shift
        
        # Apply to Oracle (Simulated Epigenetics of the Universe?)
        # No, just store it and apply in resolve_quantum_state
        pass

    # ============================================================
    # 🌍 LEVEL 6: GEO-ENGINEERING METHODS
    # ============================================================
    
    def add_structure(self, structure_info):
        """6.2 Add agent-built structure to world."""
        if structure_info is None:
            return False
        x, y = structure_info["x"], structure_info["y"]
        struct_type = structure_info["type"]
        builder_id = structure_info["builder"]
        
        if (x, y) in self.structures:
            return False  # Already occupied
        
        if struct_type == "trap":
            self.structures[(x, y)] = Trap(x, y, builder_id, 
                                            structure_info.get("harvest_rate", 0.2))
        elif struct_type == "barrier":
            self.structures[(x, y)] = Barrier(x, y, builder_id)
        elif struct_type == "battery":
            self.structures[(x, y)] = Battery(x, y, builder_id)
        elif struct_type == "cultivator":
            self.structures[(x, y)] = Cultivator(x, y, builder_id)
            self._update_cultivator_map()
        else:
            self.structures[(x, y)] = Structure(x, y, struct_type, builder_id)
        
        self.structures[(x, y)].created_tick = self.time_step
        return True
    
    def add_terrain_modification(self, terraform_info):
        """6.7 Add permanent terrain change."""
        if terraform_info is None:
            return False
        x, y = terraform_info["x"], terraform_info["y"]
        terrain = terraform_info["terrain"]
        modifier_id = terraform_info.get("modifier", "unknown")
        
        self.terrain_modifications[(x, y)] = TerrainModification(x, y, terrain, modifier_id)
        return True
    
    def _update_cultivator_map(self):
        """Update spawn boost map from cultivators."""
        self.cultivator_map = {}
        for (x, y), struct in self.structures.items():
            if isinstance(struct, Cultivator):
                for tile in struct.get_influenced_tiles(self.size):
                    current = self.cultivator_map.get(tile, 0.0)
                    self.cultivator_map[tile] = min(1.0, current + struct.boost_strength)
    
    def spawn_resource_with_cultivation(self):
        """6.5 Spawn with cultivator influence."""
        x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
        
        # Check terrain modifier
        terrain_mod = 1.0
        if (x, y) in self.terrain_modifications:
            terrain_mod = self.terrain_modifications[(x, y)].get_spawn_modifier()
        
        # Check cultivator boost
        cultivator_boost = self.cultivator_map.get((x, y), 0.0)
        
        # Effective spawn probability
        spawn_prob = min(1.0, terrain_mod * (1 + cultivator_boost))
        
        if random.random() < spawn_prob and (x, y) not in self.grid:
            self.grid[(x, y)] = Resource(x, y)
            return True
        return False
    
    def create_or_join_network(self, agent, structure_pos):
        """6.9 Create or join infrastructure network."""
        # Find nearby connected structures
        nearby_networks = set()
        for net_id, net in self.networks.items():
            for (sx, sy) in net.structures:
                if abs(sx - structure_pos[0]) <= 3 and abs(sy - structure_pos[1]) <= 3:
                    nearby_networks.add(net_id)
        
        if nearby_networks:
            # Join existing network
            net_id = list(nearby_networks)[0]
            self.networks[net_id].add_structure(*structure_pos)
            self.networks[net_id].add_member(agent.id)
            agent.join_network(net_id)
        else:
            # Create new network
            net_id = f"net_{len(self.networks)}_{self.time_step}"
            self.networks[net_id] = InfrastructureNetwork(net_id)
            self.networks[net_id].add_structure(*structure_pos)
            self.networks[net_id].add_member(agent.id)
            agent.join_network(net_id)
        
        return net_id
    
    def process_structures(self):
        """6.x Process all structures each tick."""
        to_remove = []
        for (x, y), struct in self.structures.items():
            # Decay
            if not struct.decay(0.05):
                to_remove.append((x, y))
                continue
            
            # Process traps
            if isinstance(struct, Trap):
                for agent in self.agents.values():
                    if agent.x == x and agent.y == y:
                        struct.harvest(agent)
        
            # Level 6.8: Battery Interactions (Tiny Fix)
            if isinstance(struct, Battery):
                for agent in self.agents.values():
                    if agent.x == x and agent.y == y:
                        if agent.energy > 150: # Surplus
                            struct.deposit(agent, 20)
                        elif agent.energy < 50: # Need
                            struct.withdraw(agent)
        
        # Level 6.7: Update terrain aging
        for mod in self.terrain_modifications.values():
            mod.update_age()
        
        for pos in to_remove:
            del self.structures[pos]
    
    def update_weather_control(self):
        """6.6 Collective weather control via agent votes."""
        if not self.agents:
            return
        
        total_vote = sum(a.weather_vote for a in self.agents.values() 
                         if hasattr(a, 'weather_vote'))
        n_agents = len(self.agents)
        
        # Amplitude modification (gradual change)
        target_amp = 1.0 + (total_vote / (n_agents + 1e-8)) * 0.5
        self.weather_amplitude = self.weather_amplitude * 0.95 + target_amp * 0.05
    
    def compute_env_mastery_global(self):
        """6.10 Calculate collective environmental mastery."""
        if not self.agents:
            return 0.0
        
        masteries = [a.env_control_score for a in self.agents.values() 
                     if hasattr(a, 'env_control_score')]
        return np.mean(masteries) if masteries else 0.0

    # ============================================================
    # 🐝 LEVEL 7: COLLECTIVE MANIFOLD METHODS
    # ============================================================
    
    def kuramoto_global_step(self):
        """7.1 Kuramoto Synchronization: Global phase update."""
        if len(self.agents) < 2:
            self.kuramoto_order_parameter = 0.0
            return
        
        # Update each agent's phase based on neighbors
        for agent in self.agents.values():
            if not hasattr(agent, 'kuramoto_phase'):
                continue
            neighbors = [a for a in self.agents.values() 
                         if a.id != agent.id and 
                         abs(a.x - agent.x) <= 3 and abs(a.y - agent.y) <= 3]
            agent.kuramoto_update(neighbors)
        
        # Calculate order parameter: r = |<e^{iθ}>|
        phases = [a.kuramoto_phase for a in self.agents.values() 
                  if hasattr(a, 'kuramoto_phase')]
        if phases:
            complex_order = np.mean([np.exp(1j * p) for p in phases])
            self.kuramoto_order_parameter = abs(complex_order)
    
    def federated_gradient_step(self):
        """7.2 Gradient Sharing: Pool and average gradients."""
        # Collect gradients from all agents
        self.gradient_pool = []
        for agent in self.agents.values():
            if hasattr(agent, 'last_shared_gradient') and agent.last_shared_gradient is not None:
                self.gradient_pool.append(agent.last_shared_gradient)
        
        # Share with nearby agents
        if self.time_step % 5 == 0:  # Every 5 ticks
            for agent in list(self.agents.values())[:10]:  # Limit for performance
                partners = [a for a in self.agents.values() 
                            if a.id != agent.id and 
                            abs(a.x - agent.x) <= 2 and abs(a.y - agent.y) <= 2][:3]
                if partners and hasattr(agent, 'share_gradients'):
                    agent.share_gradients(partners)
    
    def update_cognitive_modules(self):
        """7.4 Modular Cognition: Track specialist assignments."""
        self.cognitive_modules = {k: [] for k in self.cognitive_modules}
        
        for agent in self.agents.values():
            if hasattr(agent, 'cognitive_specialty') and agent.cognitive_specialty:
                if agent.cognitive_specialty in self.cognitive_modules:
                    self.cognitive_modules[agent.cognitive_specialty].append(agent.id)
    
    def process_consensus(self, proposal_id):
        """7.6 Consensus Mechanisms: Byzantine fault-tolerant voting."""
        if proposal_id not in self.consensus_registry:
            self.consensus_registry[proposal_id] = {"votes": [], "result": None}
        
        votes = []
        for agent in self.agents.values():
            if hasattr(agent, 'consensus_votes') and proposal_id in agent.consensus_votes:
                votes.append(agent.consensus_votes[proposal_id])
        
        self.consensus_registry[proposal_id]["votes"] = votes
        
        if len(votes) > len(self.agents) * 0.5:  # Quorum
            result = np.median(votes) if votes else 0
            self.consensus_registry[proposal_id]["result"] = result
            return result
        return None
    
    def update_protocol_convergence(self):
        """7.9 Emergent Protocols: Measure protocol alignment."""
        if len(self.agents) < 2:
            self.protocol_convergence = 0.0
            return
        
        protocols = [a.protocol_version for a in self.agents.values() 
                     if hasattr(a, 'protocol_version')]
        
        if len(protocols) < 2:
            self.protocol_convergence = 0.0
            return
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(protocols)):
            for j in range(i+1, len(protocols)):
                sim = 1.0 - np.mean(np.abs(protocols[i] - protocols[j]))
                similarities.append(sim)
        
        self.protocol_convergence = np.mean(similarities) if similarities else 0.0
    
    def compute_hive_phi(self):
        """7.10 Hive Mind: Calculate collective Φ."""
        total_phi = 0.0
        for agent in self.agents.values():
            if hasattr(agent, 'compute_hive_contribution'):
                total_phi += agent.compute_hive_contribution(list(self.agents.values()))
        self.hive_phi = total_phi
        return total_phi

    # ============================================================
    # 💭 LEVEL 8: ABSTRACT REPRESENTATION METHODS
    # ============================================================
    
    def update_consciousness_metrics(self):
        """8.6-8.10 Track consciousness-related metrics across population."""
        phi_values = []
        conscious_count = 0
        loop_count = 0
        
        for agent in list(self.agents.values()):
            if hasattr(agent, 'compute_phi'):
                phi = agent.compute_phi()
                phi_values.append(phi)
            
            if hasattr(agent, 'consciousness_verified') and agent.consciousness_verified:
                conscious_count += 1
            
            if hasattr(agent, 'strange_loop_active') and agent.strange_loop_active:
                loop_count += 1
        
        self.population_phi = np.mean(phi_values) if phi_values else 0.0
        self.consciousness_count = conscious_count
        self.strange_loop_count = loop_count
    
    def register_qualia(self, experience_type, activation):
        """8.9 Register global qualia prototype."""
        if experience_type not in self.qualia_registry:
            self.qualia_registry[experience_type] = activation.clone().detach() if hasattr(activation, 'clone') else activation
        else:
            # Running average
            old = self.qualia_registry[experience_type]
            if hasattr(old, 'clone') and hasattr(activation, 'clone'):
                self.qualia_registry[experience_type] = old * 0.9 + activation.clone().detach() * 0.1

    # ============================================================
    # ⚛️ LEVEL 9: UNIVERSAL HARMONIC RESONANCE METHODS
    # ============================================================
    
    def update_physics_discovery(self):
        """9.0-9.10 Track collective physics understanding."""
        best_oracle_acc = 0.0
        max_awareness = 0.0
        all_patterns = []
        
        for agent in self.agents.values():
            if hasattr(agent, 'oracle_model_accuracy'):
                best_oracle_acc = max(best_oracle_acc, agent.oracle_model_accuracy)
            
            if hasattr(agent, 'simulation_awareness'):
                max_awareness = max(max_awareness, agent.simulation_awareness)
            
            if hasattr(agent, 'discovered_patterns'):
                all_patterns.extend(agent.discovered_patterns)
            
            # Merge causal graphs
            if hasattr(agent, 'causal_bayesian_network'):
                for cause, effects in agent.causal_bayesian_network.items():
                    if cause not in self.causal_graph_collective:
                        self.causal_graph_collective[cause] = {}
                    for effect, strength in effects.items():
                        current = self.causal_graph_collective[cause].get(effect, 0)
                        self.causal_graph_collective[cause][effect] = current + strength
        
        self.collective_oracle_model_accuracy = best_oracle_acc
        self.collective_simulation_awareness = max_awareness
        self.discovered_physics_patterns = list(set(all_patterns))
        
        # Merge exploits
        all_exploits = []
        for agent in self.agents.values():
            if hasattr(agent, 'discovered_exploits'):
                all_exploits.extend(agent.discovered_exploits)
        
        # Deduplicate exploits by state_hash and action_pattern
        unique_exploits = []
        seen_exploits = set()
        for e in all_exploits:
            key = (e.get('state_hash'), e.get('action_pattern'))
            if key not in seen_exploits:
                unique_exploits.append(e)
                seen_exploits.add(key)
        self.discovered_physics_exploits = unique_exploits
        
        # 9.2 Discovery Log Update
        if not hasattr(self, 'discovery_log'):
             self.discovery_log = []
             
        # Check for new patterns to log
        known_patterns = set(entry['Pattern'] for entry in self.discovery_log)
        for p in self.discovered_physics_patterns:
            if p not in known_patterns:
                self.discovery_log.append({
                    'Time': self.time_step,
                    'Pattern': p
                })
        
        # 🔧 MEMORY FIX: Cap discovery log
        if len(self.discovery_log) > 100:
            self.discovery_log = self.discovery_log[-100:]
        
        # Check for new exploits to log
        for e in self.discovered_physics_exploits:
            pattern_name = f"exploit_{e.get('state_hash', 0) % 1000}"
            if pattern_name not in known_patterns:
                self.discovery_log.append({
                    'Time': self.time_step,
                    'Pattern': pattern_name
                })
                known_patterns.add(pattern_name)
        
        # 🔧 MEMORY FIX: Cap physics patterns
        if len(self.discovered_physics_patterns) > 50:
            self.discovered_physics_patterns = self.discovered_physics_patterns[-50:]

    # ============================================================
    # ♾️ LEVEL 10: THE OMEGA POINT METHODS
    # ============================================================
    
    def update_omega_tracking(self):
        """10.0-10.10 Track Omega Point progress."""
        self.omega_candidates = []
        max_depth = 0
        total_activity = 0
        
        for agent in self.agents.values():
            if hasattr(agent, 'scratchpad_writes'):
                total_activity += agent.scratchpad_writes
            
            if hasattr(agent, 'simulation_depth'):
                max_depth = max(max_depth, agent.simulation_depth)
            
            if hasattr(agent, 'omega_verified') and agent.omega_verified:
                self.omega_candidates.append(agent.id)
                self.omega_achieved = True
        
        self.global_scratchpad_activity = total_activity
        self.nested_simulation_depth_max = max_depth
    
    def verify_global_omega(self):
        """10.10 Global Omega Point verification."""
        criteria = {
            'multiple_conscious': self.consciousness_count >= 3,
            'high_sync': self.kuramoto_order_parameter > 0.8,
            'physics_mastery': self.collective_oracle_model_accuracy > 0.9,
            'nested_simulations': self.nested_simulation_depth_max >= 2,
            'scratchpad_active': self.global_scratchpad_activity > 100,
            'hive_mind': self.hive_phi > 5.0
        }
        
        score = sum(criteria.values()) / len(criteria)
        self.omega_achieved = score >= 0.8
        
        return {
            'achieved': self.omega_achieved,
            'score': score,
            'criteria': criteria
        }
    
    def level_6_10_step(self):
        """Combined step function for all Level 6-10 features."""
        # Level 6: Geo-Engineering
        self.process_structures()
        self.update_weather_control()
        if self.time_step % 3 == 0:
            self.spawn_resource_with_cultivation()
        
        # Level 7: Collective Manifold
        self.kuramoto_global_step()
        self.federated_gradient_step()
        self.update_cognitive_modules()
        if self.time_step % 20 == 0:
            self.update_protocol_convergence()
        if self.time_step % 10 == 0:
            self.compute_hive_phi()
        
        # Level 8: Abstract Representation (lighter, every 5 ticks)
        if self.time_step % 5 == 0:
            self.update_consciousness_metrics()
        
        # Level 9: Physics Discovery (every 10 ticks)
        if self.time_step % 10 == 0:
            self.update_physics_discovery()
        
        # Level 10: Omega Point (every 20 ticks)
        if self.time_step % 20 == 0:
            self.update_omega_tracking()
        
        # ============================================================
        # 🔧 AUDIT FIX: RUN VERIFICATION CHECKS
        # ============================================================
        if self.time_step % 100 == 0:
            self.verify_tradition_persistence()
            self.measure_cultural_drift()
            self.verify_cultural_ratchet()
            self.compute_planetary_coverage()
            self.verify_type_ii_civilization()
            self.verify_symbol_grounding()
            self._update_leadership()

    # ============================================================
    # 🔧 AUDIT FIX: NEW METHODS FOR MISSING FEATURES
    # ============================================================
    
    def verify_tradition_persistence(self):
        """3.4 Verify behavior autocorrelation > 0.7 at lag=10 generations."""
        if not self.agents:
            return False
        
        # Get current population's average behavior
        current_behaviors = []
        max_gen = max(a.generation for a in self.agents.values())
        
        for agent in self.agents.values():
            if hasattr(agent, 'tradition_history') and agent.tradition_history:
                current_behaviors.append(agent.tradition_history[-1])
        
        if not current_behaviors:
            return False
        
        # Store in tracker by generation
        self.tradition_tracker[max_gen] = current_behaviors
        
        # Clean old entries
        if len(self.tradition_tracker) > 20:
            oldest = min(self.tradition_tracker.keys())
            del self.tradition_tracker[oldest]
        
        # Check autocorrelation at lag=10
        gen_keys = sorted(self.tradition_tracker.keys())
        if len(gen_keys) < 10:
            return False
        
        behaviors_now = self.tradition_tracker.get(gen_keys[-1], [])
        behaviors_lag = self.tradition_tracker.get(gen_keys[-10], [])
        
        if not behaviors_now or not behaviors_lag:
            return False
        
        try:
            avg_now = np.mean(behaviors_now, axis=0)
            avg_lag = np.mean(behaviors_lag, axis=0)
            
            if len(avg_now) != len(avg_lag):
                return False
            
            correlation = np.corrcoef(avg_now, avg_lag)[0, 1]
            self.tradition_persistence_verified = correlation > 0.7
            return self.tradition_persistence_verified
        except:
            return False
    
    def measure_cultural_drift(self):
        """3.5 Cultural Drift: Measure KL divergence between spatial groups."""
        if len(self.agents) < 20:
            self.cultural_divergence = 0.0
            return 0.0
        
        # Divide agents into quadrants
        quadrants = {0: [], 1: [], 2: [], 3: []}
        mid = self.size // 2
        
        for agent in self.agents.values():
            q = (0 if agent.x < mid else 1) + (0 if agent.y < mid else 2)
            quadrants[q].append(agent.tag)
        
        # Calculate tag distribution divergence
        divergences = []
        for i in range(4):
            for j in range(i+1, 4):
                if quadrants[i] and quadrants[j]:
                    mean_i = np.mean(quadrants[i], axis=0)
                    mean_j = np.mean(quadrants[j], axis=0)
                    # Symmetric KL approximation
                    kl = np.sum(np.abs(mean_i - mean_j))
                    divergences.append(kl)
        
        self.cultural_divergence = np.mean(divergences) if divergences else 0.0
        return self.cultural_divergence
    
    def verify_cultural_ratchet(self):
        """3.8 Check if invention loss rate < discovery rate."""
        # Track discoveries vs losses
        current_inventions = sum(len(getattr(a, 'inventions', [])) for a in self.agents.values())
        
        if not hasattr(self, '_last_invention_count'):
            self._last_invention_count = current_inventions
            return False
        
        delta = current_inventions - self._last_invention_count
        self._last_invention_count = current_inventions
        
        event_type = 'discovery' if delta > 0 else 'loss' if delta < 0 else 'stable'
        self.invention_history.append({
            'type': event_type,
            'tick': self.time_step,
            'delta': delta
        })
        
        # Keep only last 50 entries (Reduced from 100 for memory)
        if len(self.invention_history) > 50:
            self.invention_history.pop(0)
        
        discoveries = sum(1 for e in self.invention_history if e['type'] == 'discovery')
        losses = sum(1 for e in self.invention_history if e['type'] == 'loss')
        
        self.cultural_ratchet_verified = discoveries > losses
        return self.cultural_ratchet_verified
    
    def _update_leadership(self):
        """4.9 Update alpha/leadership based on influence."""
        if not self.agents:
            self.current_alphas = []
            return
        
        agents_list = list(self.agents.values())
        agents_list.sort(key=lambda a: getattr(a, 'influence', 0), reverse=True)
        
        new_alphas = []
        for agent in agents_list[:3]:
            agent.is_alpha = True
            new_alphas.append(agent.id)
        
        for agent in agents_list[3:]:
            if hasattr(agent, 'is_alpha'):
                agent.is_alpha = False
        
        # Check for leadership turnover
        if self.current_alphas and set(new_alphas) != set(self.current_alphas):
            # Leadership changed - log event
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'event_log'):
                st.session_state.event_log.insert(0, {
                    "Tick": self.time_step,
                    "Agent": "HIVE",
                    "Event": f"👑 LEADERSHIP TURNOVER",
                    "Vector": [0]*21
                })
        
        self.current_alphas = new_alphas
    
    def compute_planetary_coverage(self):
        """6.9 Calculate structure coverage."""
        total_tiles = self.size * self.size
        structure_tiles = len(self.structures)
        self.planetary_structure_coverage = structure_tiles / total_tiles
        self.planetary_engineering_verified = self.planetary_structure_coverage > 0.01
        return self.planetary_engineering_verified
    
    def verify_type_ii_civilization(self):
        """6.10 Check if >50% energy from structures."""
        total_agent_energy = sum(a.energy for a in self.agents.values())
        structure_energy = sum(
            getattr(s, 'stored_energy', 0) for s in self.structures.values()
        )
        
        total = total_agent_energy + structure_energy
        if total > 0:
            self.structure_energy_ratio = structure_energy / total
        else:
            self.structure_energy_ratio = 0.0
        
        self.type_ii_verified = self.structure_energy_ratio > 0.5
        return self.type_ii_verified
    
    def verify_symbol_grounding(self):
        """8.0 Verify concept→environment R² > 0.7."""
        if len(self.agents) < 10:
            return False
        
        concepts = []
        env_states = []
        
        for agent in list(self.agents.values())[:30]:
            if hasattr(agent, 'last_concepts') and agent.last_concepts is not None:
                try:
                    c = agent.last_concepts.detach().cpu().numpy().flatten()
                    concepts.append(c)
                    env_sig = self.get_local_signal(agent.x, agent.y)
                    env_states.append(env_sig.numpy())
                except:
                    pass
        
        if len(concepts) < 10:
            return False
        
        try:
            X = np.array(concepts)
            Y = np.array(env_states)
            
            correlations = []
            min_dim = min(X.shape[1], Y.shape[1])
            for i in range(min_dim):
                corr = np.corrcoef(X[:, i], Y[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr ** 2)
            
            self.symbol_grounding_r2 = np.mean(correlations) if correlations else 0.0
            self.symbol_grounding_verified = self.symbol_grounding_r2 > 0.7
            return self.symbol_grounding_verified
        except:
            return False
    
    def handle_agent_death(self, dying_agent):
        """1.9 Apoptotic Information Transfer: Handle death broadcasts."""
        if not hasattr(dying_agent, 'broadcast_death_packet'):
            return
        
        # Create death packet
        death_packet = dying_agent.broadcast_death_packet()
        
        # Broadcast to nearby neighbors
        neighbors = [
            a for a in self.agents.values()
            if a.id != dying_agent.id 
            and abs(a.x - dying_agent.x) <= 3 
            and abs(a.y - dying_agent.y) <= 3
        ]
        
        for neighbor in neighbors:
            if hasattr(neighbor, 'receive_death_wisdom'):
                neighbor.receive_death_wisdom(death_packet, blend_rate=0.15)
        
        # Log the event
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'event_log'):
            st.session_state.event_log.insert(0, {
                "Tick": self.time_step,
                "Agent": dying_agent.id[:8],
                "Event": f"💀 DEATH BROADCAST → {len(neighbors)} receivers",
                "Vector": [0]*21
            })


