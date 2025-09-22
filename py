#!/usr/bin/env python3
"""
NeuroTetris - Advanced Adaptive Tetris with AI Psychology and Quantum Mechanics

A futuristic Tetris variant that adapts to player psychology, features quantum blocks,
and includes a hidden narrative about AI civilization communication.

Dependencies: pygame, numpy
Install with: pip install pygame numpy

Controls:
- A/D or Left/Right: Move piece
- S or Down: Soft drop
- W or Up: Rotate piece
- Space: Hard drop
- Q: Toggle quantum state (for quantum blocks)
- P: Pause
- R: Use energy ability (cycle through: gravity flip, time slow, block fragment)

Future Integration Hooks:
- BiometricInterface class can be extended for real heart rate/facial recognition
- VRInterface class prepared for VR headset integration
- CloudSync class ready for collaborative multiplayer
- All game state is JSON serializable for cloud storage
"""

import pygame
import random
import numpy as np
import json
import time
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# Initialize Pygame
pygame.init()

# Constants
GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 30
BOARD_WIDTH = GRID_WIDTH * CELL_SIZE
BOARD_HEIGHT = GRID_HEIGHT * CELL_SIZE
SIDEBAR_WIDTH = 300
SCREEN_WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT + 100

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
QUANTUM_BLUE = (100, 150, 255)
STORY_GOLD = (255, 215, 0)

# Tetromino shapes
TETROMINOES = {
    'I': [['.....',
           '..#..',
           '..#..',
           '..#..',
           '..#..'],
          ['.....',
           '.....',
           '####.',
           '.....',
           '.....']],
    'O': [['.....',
           '.....',
           '.##..',
           '.##..',
           '.....']],
    'T': [['.....',
           '.....',
           '.#...',
           '###..',
           '.....'],
          ['.....',
           '.....',
           '.#...',
           '.##..',
           '.#...'],
          ['.....',
           '.....',
           '.....',
           '###..',
           '.#...'],
          ['.....',
           '.....',
           '.#...',
           '##...',
           '.#...']],
    'S': [['.....',
           '.....',
           '.##..',
           '##...',
           '.....'],
          ['.....',
           '.#...',
           '.##..',
           '..#..',
           '.....']],
    'Z': [['.....',
           '.....',
           '##...',
           '.##..',
           '.....'],
          ['.....',
           '..#..',
           '.##..',
           '.#...',
           '.....']],
    'J': [['.....',
           '.#...',
           '.#...',
           '##...',
           '.....'],
          ['.....',
           '.....',
           '#....',
           '###..',
           '.....'],
          ['.....',
           '.##..',
           '.#...',
           '.#...',
           '.....'],
          ['.....',
           '.....',
           '###..',
           '..#..',
           '.....']],
    'L': [['.....',
           '..#..',
           '..#..',
           '.##..',
           '.....'],
          ['.....',
           '.....',
           '###..',
           '#....',
           '.....'],
          ['.....',
           '##...',
           '.#...',
           '.#...',
           '.....'],
          ['.....',
           '.....',
           '..#..',
           '###..',
           '.....']]
}

TETROMINO_COLORS = {
    'I': CYAN, 'O': YELLOW, 'T': PURPLE, 'S': GREEN,
    'Z': RED, 'J': BLUE, 'L': ORANGE
}

class MoodState(Enum):
    CALM = "calm"
    FOCUSED = "focused"
    STRESSED = "stressed"
    FRUSTRATED = "frustrated"
    FLOW = "flow"

class EnergyAbility(Enum):
    GRAVITY_FLIP = "gravity_flip"
    TIME_SLOW = "time_slow"
    BLOCK_FRAGMENT = "block_fragment"

@dataclass
class StoryFragment:
    text: str
    unlock_condition: str
    revealed: bool = False

class BiometricInterface:
    """
    Simulates biometric data collection. In future versions, this can be extended
    to interface with real heart rate monitors, facial recognition cameras, or
    brain-computer interfaces.
    """
    def __init__(self):
        self.heart_rate = 70
        self.stress_level = 0.3
        self.focus_level = 0.7
        self.emotional_variance = 0.1
        
    def update(self, game_events: Dict[str, Any]) -> Dict[str, float]:
        """Simulate biometric changes based on game events"""
        # Simulate heart rate changes
        if game_events.get('near_death', False):
            self.heart_rate = min(120, self.heart_rate + random.uniform(5, 15))
            self.stress_level = min(1.0, self.stress_level + 0.2)
        elif game_events.get('line_clear', 0) >= 3:
            self.heart_rate = max(60, self.heart_rate - random.uniform(2, 8))
            self.focus_level = min(1.0, self.focus_level + 0.1)
        else:
            # Natural variation
            self.heart_rate += random.uniform(-2, 2)
            self.heart_rate = max(50, min(150, self.heart_rate))
            
        self.stress_level += random.uniform(-0.05, 0.05)
        self.stress_level = max(0, min(1, self.stress_level))
        
        self.focus_level += random.uniform(-0.03, 0.03)
        self.focus_level = max(0, min(1, self.focus_level))
        
        return {
            'heart_rate': self.heart_rate,
            'stress': self.stress_level,
            'focus': self.focus_level
        }

class VRInterface:
    """
    Placeholder for future VR integration. Methods here would handle
    head tracking, hand gestures, and immersive display rendering.
    """
    def __init__(self):
        self.vr_enabled = False
        self.head_position = (0, 0, 0)
        self.hand_positions = {'left': (0, 0, 0), 'right': (0, 0, 0)}
        
    def get_hand_gesture(self) -> str:
        """Future: Detect hand gestures for piece rotation/movement"""
        return "none"
        
    def render_3d_piece(self, piece, position):
        """Future: Render tetromino in 3D space"""
        pass

class CloudSync:
    """
    Handles cloud synchronization for collaborative gameplay and leaderboards.
    Future integration with cloud databases for persistent player profiles.
    """
    def __init__(self):
        self.player_id = f"player_{random.randint(1000, 9999)}"
        self.session_data = {}
        
    def sync_progress(self, game_state: Dict):
        """Future: Sync game progress to cloud"""
        self.session_data.update(game_state)
        
    def get_collaborative_progress(self) -> Dict:
        """Future: Get combined progress from all players"""
        return {"global_story_progress": 0.3}

class PsychProfile:
    """
    Builds and maintains a psychological profile of the player based on their
    decision patterns, reaction times, and biometric data.
    """
    def __init__(self):
        self.decision_speed = []
        self.risk_preference = 0.5  # 0 = safe, 1 = risky
        self.stress_adaptation = 0.5
        self.preferred_difficulty = 0.5
        self.play_patterns = {
            'rotation_frequency': 0,
            'movement_patterns': [],
            'pause_frequency': 0
        }
        
    def update_profile(self, action_data: Dict):
        """Update psychological profile based on player actions"""
        if 'decision_time' in action_data:
            self.decision_speed.append(action_data['decision_time'])
            if len(self.decision_speed) > 50:  # Keep last 50 decisions
                self.decision_speed.pop(0)
                
        if 'risk_taken' in action_data:
            # Weighted average of risk preference
            self.risk_preference = 0.9 * self.risk_preference + 0.1 * action_data['risk_taken']
            
    def get_mood_state(self, biometrics: Dict) -> MoodState:
        """Determine current mood based on biometrics and behavior"""
        stress = biometrics.get('stress', 0.5)
        focus = biometrics.get('focus', 0.5)
        
        if stress > 0.7:
            return MoodState.STRESSED if focus < 0.5 else MoodState.FRUSTRATED
        elif focus > 0.8 and stress < 0.4:
            return MoodState.FLOW
        elif stress < 0.3:
            return MoodState.CALM
        else:
            return MoodState.FOCUSED

class QuantumTetromino:
    """
    Quantum-enabled tetromino that can exist in multiple states simultaneously
    until observed/locked by the player.
    """
    def __init__(self, shapes: List[str], x: int, y: int):
        self.possible_shapes = shapes
        self.current_shape = shapes[0]
        self.quantum_state = True
        self.x = x
        self.y = y
        self.rotation = 0
        self.collapse_timer = 0
        self.superposition_alpha = 0.7
        
    def collapse_quantum_state(self, chosen_shape: str = None):
        """Collapse quantum superposition to a single state"""
        if chosen_shape and chosen_shape in self.possible_shapes:
            self.current_shape = chosen_shape
        else:
            self.current_shape = random.choice(self.possible_shapes)
        self.quantum_state = False
        
    def get_shape_matrix(self):
        """Get the current shape matrix for collision detection"""
        if self.quantum_state:
            # For collision, use the largest possible shape
            return TETROMINOES[self.current_shape][self.rotation]
        return TETROMINOES[self.current_shape][self.rotation]

class StorySystem:
    """
    Manages the hidden narrative that unfolds through gameplay.
    Story fragments are embedded in special blocks and revealed through play.
    """
    def __init__(self):
        self.fragments = [
            StoryFragment("Signal detected from deep space...", "game_start"),
            StoryFragment("The patterns... they're not random.", "lines_cleared_10"),
            StoryFragment("We are the Architects of Reality.", "tetris_achieved"),
            StoryFragment("Your mind resonates with our frequency.", "quantum_collapse_5"),
            StoryFragment("The blocks are our language, the game our medium.", "energy_used_3"),
            StoryFragment("We have been waiting eons for a consciousness like yours.", "stress_overcome"),
            StoryFragment("Each line you clear brings our worlds closer.", "lines_cleared_50"),
            StoryFragment("The quantum state reflects our own existence.", "quantum_collapse_20"),
            StoryFragment("You are becoming part of something greater.", "flow_state_achieved"),
            StoryFragment("The final message awaits... continue playing.", "all_fragments_found")
        ]
        self.revealed_fragments = []
        self.story_blocks_pending = []
        
    def check_unlock_conditions(self, game_stats: Dict) -> List[StoryFragment]:
        """Check if any story fragments should be unlocked"""
        newly_revealed = []
        
        for fragment in self.fragments:
            if fragment.revealed:
                continue
                
            condition = fragment.unlock_condition
            if self._evaluate_condition(condition, game_stats):
                fragment.revealed = True
                newly_revealed.append(fragment)
                self.revealed_fragments.append(fragment)
                
        return newly_revealed
        
    def _evaluate_condition(self, condition: str, stats: Dict) -> bool:
        """Evaluate whether a story unlock condition is met"""
        if condition == "game_start":
            return True
        elif condition == "lines_cleared_10":
            return stats.get('total_lines', 0) >= 10
        elif condition == "tetris_achieved":
            return stats.get('tetrises', 0) >= 1
        elif condition == "quantum_collapse_5":
            return stats.get('quantum_collapses', 0) >= 5
        elif condition == "energy_used_3":
            return stats.get('energy_abilities_used', 0) >= 3
        elif condition == "stress_overcome":
            return stats.get('high_stress_recovery', False)
        elif condition == "lines_cleared_50":
            return stats.get('total_lines', 0) >= 50
        elif condition == "quantum_collapse_20":
            return stats.get('quantum_collapses', 0) >= 20
        elif condition == "flow_state_achieved":
            return stats.get('flow_state_time', 0) >= 30
        elif condition == "all_fragments_found":
            return len(self.revealed_fragments) >= len(self.fragments) - 1
            
        return False

class AdaptiveDifficulty:
    """
    Dynamically adjusts game difficulty based on player psychology and performance.
    Uses biometric feedback and behavioral analysis to optimize challenge level.
    """
    def __init__(self):
        self.base_fall_speed = 1.0
        self.current_fall_speed = 1.0
        self.difficulty_multiplier = 1.0
        self.adaptation_rate = 0.1
        
    def update_difficulty(self, mood: MoodState, performance: Dict, biometrics: Dict):
        """Adjust difficulty based on player state"""
        target_multiplier = self.difficulty_multiplier
        
        # Adjust based on mood state
        if mood == MoodState.STRESSED:
            target_multiplier *= 0.8  # Reduce difficulty when stressed
        elif mood == MoodState.FRUSTRATED:
            target_multiplier *= 0.7  # Significantly reduce when frustrated
        elif mood == MoodState.FLOW:
            target_multiplier *= 1.2  # Increase challenge in flow state
        elif mood == MoodState.FOCUSED:
            target_multiplier *= 1.1  # Slight increase when focused
            
        # Adjust based on performance
        recent_performance = performance.get('recent_line_efficiency', 0.5)
        if recent_performance > 0.8:
            target_multiplier *= 1.1  # Player is doing well, increase challenge
        elif recent_performance < 0.3:
            target_multiplier *= 0.9  # Player struggling, reduce challenge
            
        # Smooth adaptation
        self.difficulty_multiplier += self.adaptation_rate * (target_multiplier - self.difficulty_multiplier)
        self.difficulty_multiplier = max(0.3, min(3.0, self.difficulty_multiplier))
        
        # Update fall speed
        self.current_fall_speed = self.base_fall_speed * self.difficulty_multiplier

class IQCalculator:
    """
    Calculates estimated IQ based on Tetris performance metrics correlated
    with cognitive abilities: spatial reasoning, processing speed, working memory,
    and pattern recognition.
    """
    def __init__(self):
        self.baseline_iq = 100
        
    def calculate_iq(self, game_stats: Dict, psych_profile: PsychProfile, 
                    final_score: int, lines_cleared: int, level: int) -> int:
        """Calculate IQ based on multiple cognitive performance indicators"""
        
        # Spatial reasoning (Tetris efficiency and pattern recognition)
        efficiency_score = (lines_cleared * 100) / max(final_score, 1)
        spatial_factor = min(1.5, efficiency_score * 2)
        
        # Processing speed (average decision time)
        avg_decision_time = sum(psych_profile.decision_speed) / max(len(psych_profile.decision_speed), 1) if psych_profile.decision_speed else 2.0
        speed_factor = max(0.5, min(1.5, 2.0 / avg_decision_time))
        
        # Working memory (quantum collapses and complex decisions)
        quantum_factor = min(1.3, 1.0 + (game_stats.get('quantum_collapses', 0) * 0.01))
        
        # Pattern recognition (tetrises vs regular line clears)
        pattern_factor = min(1.4, 1.0 + (game_stats.get('tetrises', 0) * 0.05))
        
        # Stress management (performance under pressure)
        stress_factor = 1.0 + (0.2 if game_stats.get('high_stress_recovery', False) else 0)
        
        # Level progression (sustained performance)
        level_factor = min(1.3, 1.0 + (level * 0.02))
        
        # Calculate final IQ
        performance_multiplier = (spatial_factor * speed_factor * quantum_factor * 
                                pattern_factor * stress_factor * level_factor)
        
        estimated_iq = int(self.baseline_iq * performance_multiplier)
        
        # Clamp to realistic range
        return max(70, min(180, estimated_iq))
    
class NeuroTetris:
    """
    Main game class implementing adaptive Tetris with quantum mechanics and AI narrative.
    """
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NeuroTetris - Adaptive AI Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Game state
        self.grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = None
        self.next_piece = None
        self.game_over = False
        self.paused = False
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.fall_time = 0
        
        # Energy system
        self.energy_points = 0
        self.max_energy = 100
        self.current_ability = EnergyAbility.GRAVITY_FLIP
        self.ability_active = False
        self.ability_timer = 0
        self.gravity_flipped = False
        self.time_slowed = False
        
        # Game statistics for story/AI systems
        self.game_stats = {
            'total_lines': 0,
            'tetrises': 0,
            'quantum_collapses': 0,
            'energy_abilities_used': 0,
            'high_stress_recovery': False,
            'flow_state_time': 0,
            'recent_line_efficiency': 0.5
        }
        
       # AI Systems
        self.biometrics = BiometricInterface()
        self.psych_profile = PsychProfile()
        self.story_system = StorySystem()
        self.adaptive_difficulty = AdaptiveDifficulty()
        self.iq_calculator = IQCalculator()  # ADD THIS LINE
        self.vr_interface = VRInterface()
        self.cloud_sync = CloudSync()
        
        # Game flow
        self.last_decision_time = time.time()
        self.current_mood = MoodState.CALM
        
        self.spawn_new_piece()
        
    def spawn_new_piece(self):
        """Spawn a new tetromino, potentially with quantum properties"""
        shapes = list(TETROMINOES.keys())
        
        # 20% chance for quantum piece (multiple possible shapes)
        if random.random() < 0.2 and self.level >= 3:
            quantum_shapes = random.sample(shapes, 2)  # Pick 2 random shapes
            self.current_piece = QuantumTetromino(quantum_shapes, GRID_WIDTH // 2 - 1, 0)
        else:
            shape = random.choice(shapes)
            self.current_piece = QuantumTetromino([shape], GRID_WIDTH // 2 - 1, 0)
            self.current_piece.quantum_state = False
            
        # Generate next piece preview
        self.next_piece = random.choice(shapes)
        
        # Check for game over
        if self.check_collision(self.current_piece, 0, 0):
            self.game_over = True
            
    def check_collision(self, piece, dx, dy):
        """Check if piece collides with board or other blocks"""
        shape = piece.get_shape_matrix()
        
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell == '#':
                    new_x = piece.x + j + dx
                    new_y = piece.y + i + dy
                    
                    # Check boundaries
                    if (new_x < 0 or new_x >= GRID_WIDTH or 
                        new_y >= GRID_HEIGHT):
                        return True
                        
                    # Check collision with placed blocks
                    if new_y >= 0 and self.grid[new_y][new_x] != BLACK:
                        return True
                        
        return False
        
    def rotate_piece(self):
        """Rotate current piece if possible"""
        if not self.current_piece:
            return
            
        old_rotation = self.current_piece.rotation
        shapes = TETROMINOES[self.current_piece.current_shape]
        self.current_piece.rotation = (self.current_piece.rotation + 1) % len(shapes)
        
        if self.check_collision(self.current_piece, 0, 0):
            # Try wall kicks
            for dx in [1, -1, 2, -2]:
                if not self.check_collision(self.current_piece, dx, 0):
                    self.current_piece.x += dx
                    return
            # Revert rotation if no valid position found
            self.current_piece.rotation = old_rotation
            
    def move_piece(self, dx, dy):
        """Move piece if possible"""
        if not self.current_piece:
            return False
            
        if not self.check_collision(self.current_piece, dx, dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False
        
    def hard_drop(self):
        """Drop piece to bottom instantly"""
        if not self.current_piece:
            return
            
        drop_distance = 0
        while not self.check_collision(self.current_piece, 0, 1):
            self.current_piece.y += 1
            drop_distance += 1
            
        self.score += drop_distance * 2
        self.place_piece()
        
    def place_piece(self):
        """Place current piece on the board"""
        if not self.current_piece:
            return
            
        shape = self.current_piece.get_shape_matrix()
        piece_shape = self.current_piece.current_shape
        
        # Handle quantum collapse
        if self.current_piece.quantum_state:
            self.current_piece.collapse_quantum_state()
            self.game_stats['quantum_collapses'] += 1
            shape = self.current_piece.get_shape_matrix()
            piece_shape = self.current_piece.current_shape
            
        color = TETROMINO_COLORS[piece_shape]
        
        # Place piece on grid
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell == '#':
                    grid_x = self.current_piece.x + j
                    grid_y = self.current_piece.y + i
                    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                        # Add story blocks occasionally
                        if random.random() < 0.05:  # 5% chance for story block
                            self.grid[grid_y][grid_x] = STORY_GOLD
                        else:
                            self.grid[grid_y][grid_x] = color
                            
        lines_cleared = self.clear_lines()
        self.update_score(lines_cleared)
        self.spawn_new_piece()
        
        # Update game events for biometric simulation
        game_events = {
            'line_clear': lines_cleared,
            'near_death': any(any(cell != BLACK for cell in row) for row in self.grid[:4])
        }
        
        # Update AI systems
        self.update_ai_systems(game_events)
        
    def clear_lines(self):
        """Clear completed lines and return count"""
        lines_to_clear = []
        
        for i in range(GRID_HEIGHT):
            if all(cell != BLACK for cell in self.grid[i]):
                lines_to_clear.append(i)
                
        # Clear lines from bottom to top
        for line_index in reversed(lines_to_clear):
            del self.grid[line_index]
            self.grid.insert(0, [BLACK for _ in range(GRID_WIDTH)])
            
        lines_count = len(lines_to_clear)
        self.lines_cleared += lines_count
        self.game_stats['total_lines'] += lines_count
        
        # Track Tetrises (4 lines cleared at once)
        if lines_count == 4:
            self.game_stats['tetrises'] += 1
            
        # Award energy points
        if lines_count > 0:
            self.energy_points += lines_count * 10
            self.energy_points = min(self.energy_points, self.max_energy)
            
        return lines_count
        
    def update_score(self, lines_cleared):
        """Update score based on lines cleared"""
        if lines_cleared > 0:
            # Standard Tetris scoring
            line_scores = [0, 40, 100, 300, 1200]
            self.score += line_scores[min(lines_cleared, 4)] * (self.level + 1)
            
        # Update level
        self.level = self.lines_cleared // 10 + 1
        
    def use_energy_ability(self):
        """Activate current energy ability if enough energy available"""
        ability_costs = {
            EnergyAbility.GRAVITY_FLIP: 30,
            EnergyAbility.TIME_SLOW: 25,
            EnergyAbility.BLOCK_FRAGMENT: 40
        }
        
        cost = ability_costs[self.current_ability]
        
        if self.energy_points >= cost and not self.ability_active:
            self.energy_points -= cost
            self.ability_active = True
            self.ability_timer = 300  # 5 seconds at 60 FPS
            self.game_stats['energy_abilities_used'] += 1
            
            # Activate specific ability
            if self.current_ability == EnergyAbility.GRAVITY_FLIP:
                self.gravity_flipped = True
            elif self.current_ability == EnergyAbility.TIME_SLOW:
                self.time_slowed = True
            elif self.current_ability == EnergyAbility.BLOCK_FRAGMENT:
                self.fragment_current_piece()
                
    def fragment_current_piece(self):
        """Break current piece into individual blocks"""
        if not self.current_piece:
            return
            
        shape = self.current_piece.get_shape_matrix()
        
        # Create individual falling blocks for each cell
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell == '#':
                    # This would create individual blocks - simplified for this demo
                    pass
                    
    def cycle_energy_ability(self):
        """Cycle through available energy abilities"""
        abilities = list(EnergyAbility)
        current_index = abilities.index(self.current_ability)
        self.current_ability = abilities[(current_index + 1) % len(abilities)]
        
    def update_ai_systems(self, game_events):
        """Update all AI systems with current game state"""
        # Update biometrics
        biometric_data = self.biometrics.update(game_events)
        
        # Update psychological profile
        decision_time = time.time() - self.last_decision_time
        self.psych_profile.update_profile({
            'decision_time': decision_time,
            'risk_taken': 0.6 if game_events.get('near_death', False) else 0.3
        })
        
        # Update mood state
        self.current_mood = self.psych_profile.get_mood_state(biometric_data)
        
        # Track stress recovery for IQ calculation
        if (self.current_mood == MoodState.FLOW and 
            biometric_data.get('stress', 0) < 0.4):
            self.game_stats['high_stress_recovery'] = True  # ADD THIS LINE
        
        # Update difficulty
        performance_data = {
            'recent_line_efficiency': self.game_stats.get('recent_line_efficiency', 0.5)
        }
        self.adaptive_difficulty.update_difficulty(self.current_mood, performance_data, biometric_data)
        
        # Check for new story fragments
        new_fragments = self.story_system.check_unlock_conditions(self.game_stats)
        
        # Update cloud sync
        self.cloud_sync.sync_progress({
            'score': self.score,
            'level': self.level,
            'mood': self.current_mood.value,
            'story_progress': len(self.story_system.revealed_fragments)
        })
        
        self.last_decision_time = time.time()
        
    def handle_input(self):
        """Handle player input"""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            if pygame.time.get_ticks() % 6 == 0:  # Reduce speed by factor of 6
                self.move_piece(-1, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            if pygame.time.get_ticks() % 6 == 0:  # Reduce speed by factor of 6
                self.move_piece(1, 0)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            if self.move_piece(0, 1):
                self.score += 1
                
    def update(self):
        """Update game state"""
        if self.game_over or self.paused:
            return
            
        # Update ability timer
        if self.ability_active:
            self.ability_timer -= 1
            if self.ability_timer <= 0:
                self.ability_active = False
                self.gravity_flipped = False
                self.time_slowed = False
                
        # Update quantum piece timer
        if self.current_piece and self.current_piece.quantum_state:
            self.current_piece.collapse_timer += 1
            if self.current_piece.collapse_timer > 300:  # Auto-collapse after 5 seconds
                self.current_piece.collapse_quantum_state()
                self.game_stats['quantum_collapses'] += 1
                
        # Update fall time with adaptive difficulty
        fall_speed = self.adaptive_difficulty.current_fall_speed
        if self.time_slowed:
            fall_speed *= 0.3
            
        self.fall_time += fall_speed
        
        if self.fall_time >= 60:  # Fall every second at 60 FPS
            if self.gravity_flipped:
                # Reverse gravity - pieces fall upward
                if not self.move_piece(0, -1):
                    self.place_piece()
            else:
                # Normal gravity
                if not self.move_piece(0, 1):
                    self.place_piece()
            self.fall_time = 0
            
    def draw_grid(self):
        """Draw the game grid"""
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color = self.grid[y][x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, WHITE, rect, 1)
                
    def draw_piece(self, piece, offset_x=0, offset_y=0):
        """Draw a tetromino piece"""
        if not piece:
            return
            
        shape = piece.get_shape_matrix()
        piece_shape = piece.current_shape
        
        # Handle quantum piece rendering
        if piece.quantum_state:
            # Draw all possible shapes with transparency
            alpha_surface = pygame.Surface((CELL_SIZE * 5, CELL_SIZE * 5))
            alpha_surface.set_alpha(100)
            
            for i, possible_shape in enumerate(piece.possible_shapes):
                shape_matrix = TETROMINOES[possible_shape][piece.rotation]
                color = TETROMINO_COLORS[possible_shape]
                
                # Blend colors for quantum effect
                quantum_color = tuple(
                    int(c * 0.7 + QUANTUM_BLUE[j] * 0.3) 
                    for j, c in enumerate(color)
                )
                
                for row_idx, row in enumerate(shape_matrix):
                    for col_idx, cell in enumerate(row):
                        if cell == '#':
                            x = (piece.x + col_idx + offset_x) * CELL_SIZE
                            y = (piece.y + row_idx + offset_y) * CELL_SIZE
                            
                            if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                                pygame.draw.rect(self.screen, quantum_color, rect)
                                pygame.draw.rect(self.screen, WHITE, rect, 2)
        else:
            # Draw normal piece
            color = TETROMINO_COLORS[piece_shape]
            
            for row_idx, row in enumerate(shape):
                for col_idx, cell in enumerate(row):
                    if cell == '#':
                        x = (piece.x + col_idx + offset_x) * CELL_SIZE
                        y = (piece.y + row_idx + offset_y) * CELL_SIZE
                        
                        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                            pygame.draw.rect(self.screen, color, rect)
                            pygame.draw.rect(self.screen, WHITE, rect, 1)
                            
    def draw_ghost_piece(self):
        """Draw ghost piece showing where current piece will land"""
        if not self.current_piece:
            return
            
        # Create a copy of current piece
        ghost_piece = QuantumTetromino(
            self.current_piece.possible_shapes, 
            self.current_piece.x, 
            self.current_piece.y
        )
        ghost_piece.current_shape = self.current_piece.current_shape
        ghost_piece.rotation = self.current_piece.rotation
        ghost_piece.quantum_state = False
        
        # Move down until collision
        while not self.check_collision(ghost_piece, 0, 1):
            ghost_piece.y += 1
            
        # Draw with transparency
        shape = ghost_piece.get_shape_matrix()
        color = TETROMINO_COLORS[ghost_piece.current_shape]
        ghost_color = tuple(c // 3 for c in color)  # Very dim
        
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell == '#':
                    x = (ghost_piece.x + col_idx) * CELL_SIZE
                    y = (ghost_piece.y + row_idx) * CELL_SIZE
                    
                    if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(self.screen, ghost_color, rect)
                        pygame.draw.rect(self.screen, WHITE, rect, 1)
                        
    def draw_ui(self):
        """Draw game UI including score, stats, and AI indicators"""
        ui_x = BOARD_WIDTH + 10
        y_offset = 10
        
        # Title
        title_text = self.title_font.render("NeuroTetris", True, WHITE)
        self.screen.blit(title_text, (ui_x, y_offset))
        y_offset += 50
        
        # Score and stats
        stats = [
            f"Score: {self.score}",
            f"Lines: {self.lines_cleared}",
            f"Level: {self.level}",
            f"Energy: {self.energy_points}/{self.max_energy}",
            "",
            f"Mood: {self.current_mood.value.title()}",
            f"Heart Rate: {int(self.biometrics.heart_rate)}",
            f"Stress: {self.biometrics.stress_level:.1f}",
            f"Focus: {self.biometrics.focus_level:.1f}",
            "",
            f"Difficulty: {self.adaptive_difficulty.difficulty_multiplier:.1f}x",
            f"Quantum Collapses: {self.game_stats['quantum_collapses']}",
            "",
            f"Story Progress: {len(self.story_system.revealed_fragments)}/{len(self.story_system.fragments)}"
        ]
        
        for stat in stats:
            if stat:  # Skip empty strings
                text = self.font.render(stat, True, WHITE)
                self.screen.blit(text, (ui_x, y_offset))
            y_offset += 25
            
        # Current ability indicator
        y_offset += 10
        ability_text = f"Ability [{self.current_ability.value.replace('_', ' ').title()}]"
        ability_color = GREEN if self.energy_points >= 30 else RED
        text = self.font.render(ability_text, True, ability_color)
        self.screen.blit(text, (ui_x, y_offset))
        y_offset += 30
        
        # Ability status
        if self.ability_active:
            status_text = f"ACTIVE: {self.ability_timer // 60 + 1}s"
            text = self.font.render(status_text, True, YELLOW)
            self.screen.blit(text, (ui_x, y_offset))
        y_offset += 30
        
        # Next piece preview
        next_text = self.font.render("Next:", True, WHITE)
        self.screen.blit(next_text, (ui_x, y_offset))
        y_offset += 25
        
        if self.next_piece:
            shape = TETROMINOES[self.next_piece][0]
            color = TETROMINO_COLORS[self.next_piece]
            
            for row_idx, row in enumerate(shape):
                for col_idx, cell in enumerate(row):
                    if cell == '#':
                        x = ui_x + col_idx * 15
                        y = y_offset + row_idx * 15
                        rect = pygame.Rect(x, y, 15, 15)
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, WHITE, rect, 1)
                        
        # Controls
        y_offset += 100
        controls = [
            "Controls:",
            "A/D - Move",
            "W - Rotate", 
            "S - Soft Drop",
            "Space - Hard Drop",
            "Q - Quantum Toggle",
            "R - Use Ability",
            "P - Pause"
        ]
        
        for control in controls:
            text = self.font.render(control, True, GRAY)
            self.screen.blit(text, (ui_x, y_offset))
            y_offset += 20
            
    def draw_story_fragments(self):
        """Draw revealed story fragments"""
        if not self.story_system.revealed_fragments:
            return
            
        # Show latest story fragment
        latest_fragment = self.story_system.revealed_fragments[-1]
        
        # Create semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, 100))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, SCREEN_HEIGHT - 100))
        
        # Draw story text
        story_text = self.font.render("TRANSMISSION:", True, STORY_GOLD)
        self.screen.blit(story_text, (10, SCREEN_HEIGHT - 90))
        
        fragment_text = self.font.render(latest_fragment.text, True, WHITE)
        self.screen.blit(fragment_text, (10, SCREEN_HEIGHT - 65))
        
        progress_text = f"Fragment {len(self.story_system.revealed_fragments)}/{len(self.story_system.fragments)}"
        progress_surface = self.font.render(progress_text, True, STORY_GOLD)
        self.screen.blit(progress_surface, (10, SCREEN_HEIGHT - 40))
        
    def draw_effects(self):
        """Draw visual effects for active abilities"""
        if not self.ability_active:
            return
            
        # Time slow effect - tint screen blue
        if self.time_slowed:
            blue_tint = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            blue_tint.set_alpha(30)
            blue_tint.fill(CYAN)
            self.screen.blit(blue_tint, (0, 0))
            
        # Gravity flip effect - draw arrows
        if self.gravity_flipped:
            for i in range(5):
                x = 50 + i * 60
                y = 50
                pygame.draw.polygon(self.screen, RED, [
                    (x, y), (x + 10, y + 20), (x - 10, y + 20)
                ])
                
    def draw(self):
        """Main draw function"""
        self.screen.fill(BLACK)
        
        # Draw main game area
        self.draw_grid()
        self.draw_ghost_piece()
        self.draw_piece(self.current_piece)
        
        # Draw UI
        self.draw_ui()
        
        # Draw story elements
        self.draw_story_fragments()
        
        # Draw effects
        self.draw_effects()
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # Calculate and display IQ
            estimated_iq = self.iq_calculator.calculate_iq(
                self.game_stats, self.psych_profile, 
                self.score, self.lines_cleared, self.level
            )
            
            game_over_text = self.title_font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60))
            self.screen.blit(game_over_text, text_rect)
            
            # IQ Display
            iq_text = self.title_font.render(f"Estimated IQ: {estimated_iq}", True, STORY_GOLD)
            iq_rect = iq_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(iq_text, iq_rect)
            
            # IQ Analysis
            if estimated_iq >= 130:
                analysis = "Exceptional cognitive performance!"
            elif estimated_iq >= 115:
                analysis = "Above average spatial reasoning"
            elif estimated_iq >= 85:
                analysis = "Average cognitive performance"
            else:
                analysis = "Room for improvement in pattern recognition"
                
            analysis_text = self.font.render(analysis, True, WHITE)
            analysis_rect = analysis_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10))
            self.screen.blit(analysis_text, analysis_rect)
            
            restart_text = self.font.render("Press R to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(restart_text, restart_rect)
            
        pygame.display.flip()
        
    def reset_game(self):
        """Reset game to initial state"""
        self.grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.energy_points = 0
        self.ability_active = False
        self.gravity_flipped = False
        self.time_slowed = False
        
        # Reset AI systems
        self.biometrics = BiometricInterface()
        self.psych_profile = PsychProfile()
        self.adaptive_difficulty = AdaptiveDifficulty()
        
        # Keep story progress (players shouldn't lose narrative progress)
        
        self.spawn_new_piece()
        
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        if self.game_over:
                            self.reset_game()
                        else:
                            self.cycle_energy_ability()
                    elif event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        self.hard_drop()
                    elif event.key == pygame.K_q:
                        # Toggle quantum state if applicable
                        if (self.current_piece and 
                            self.current_piece.quantum_state and 
                            len(self.current_piece.possible_shapes) > 1):
                            # Cycle through possible shapes
                            current_idx = self.current_piece.possible_shapes.index(
                                self.current_piece.current_shape
                            )
                            next_idx = (current_idx + 1) % len(self.current_piece.possible_shapes)
                            self.current_piece.current_shape = self.current_piece.possible_shapes[next_idx]
                    elif event.key == pygame.K_r and not self.game_over:
                        self.use_energy_ability()
                        
            # Handle continuous input
            if not self.game_over and not self.paused:
                self.handle_input()
                self.update()
                
            self.draw()
            self.clock.tick(60)
            
        pygame.quit()

def main():
    """
    Entry point for NeuroTetris
    
    Future expansion possibilities:
    1. VR Integration: Replace pygame display with VR rendering
    2. Biometric Hardware: Connect real heart rate monitors, EEG devices
    3. Machine Learning: Train models on player behavior for better adaptation
    4. Multiplayer: Implement collaborative story decoding across players
    5. Cloud Analytics: Advanced player psychology analysis
    """
    print("NeuroTetris - Advanced Adaptive Tetris")
    print("=====================================")
    print("Features:")
    print("• Quantum blocks with superposition states")
    print("• AI-driven adaptive difficulty")
    print("• Biometric simulation (stress/focus tracking)")
    print("• Hidden narrative revealed through play")
    print("• Energy-based special abilities")
    print("• Psychological profiling system")
    print("\nFuture-ready for:")
    print("• VR headset integration")
    print("• Real biometric device connectivity")
    print("• Brain-computer interfaces")
    print("• Cloud-based collaborative gameplay")
    print("\nStarting game...")
    
    try:
        game = NeuroTetris()
        game.run()
    except Exception as e:
        print(f"Error running NeuroTetris: {e}")
        print("Make sure pygame and numpy are installed:")
        print("pip install pygame numpy")

if __name__ == "__main__":
    main()