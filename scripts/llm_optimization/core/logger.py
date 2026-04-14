"""
Logger infrastructure for per-step state logging and metric computation.

This module provides the base Logger class and game-specific implementations
for the current unified thesis regime: provided metrics plus full state access
for Pong, Freeway, and Asterix.

Full State Access:
- Pong: player_y, player_speed, ball_x, ball_y, ball_vel_x, ball_vel_y, enemy_y, scores
- Freeway: chicken_y, cars (all positions), score, cooldown, lives_lost

Provided Metrics:
- Pong: win_rate, ball_tracking_error, paddle_efficiency, rally_length, reaction_time
- Freeway: crossings, collision_rate, upward_progress, lane_transitions, near_misses
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, NamedTuple
import jax.numpy as jnp
import numpy as np


@dataclass
class AblationConfig:
    """Configuration for ablation study settings.
    
    Axis A: Metrics Source
        A1: Metrics are provided (predefined by humans)
        A2: LLM self-proposes the metrics to track
    
    Axis B: Information Access
        B1: Full game state (RAM, internal state variables)
        B2: Observables only (what the policy can observe)
    """
    metrics_source: str = "A1"  # "A1" (provided) or "A2" (self-proposed)
    info_access: str = "B1"     # "B1" (full state) or "B2" (observables only)
    
    @property
    def setting_name(self) -> str:
        """Return the combined setting name (e.g., 'A1_B1')."""
        return f"{self.metrics_source}_{self.info_access}"
    
    def __post_init__(self):
        if self.metrics_source not in ["A1", "A2"]:
            raise ValueError(f"metrics_source must be 'A1' or 'A2', got {self.metrics_source}")
        if self.info_access not in ["B1", "B2"]:
            raise ValueError(f"info_access must be 'B1' or 'B2', got {self.info_access}")


class Logger(ABC):
    """Abstract base class for per-step logging and metric computation.
    
    The Logger is responsible for:
    1. Receiving state every step via log_state()
    2. Computing metrics at the end of the episode via return_metrics()
    3. Providing metrics to the LLM for policy improvement feedback
    
    Subclasses should implement game-specific logging and metric computation.
    """
    
    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        """Initialize the logger.
        
        Args:
            ablation_config: Configuration specifying A1/A2 and B1/B2 settings.
                           If None, defaults to A1_B1 (provided metrics, full state).
        """
        self.config = ablation_config or AblationConfig()
        self._episode_data: List[Dict[str, Any]] = []
        self._episode_count = 0
        
    def reset(self):
        """Reset the logger for a new episode."""
        self._episode_data = []
        self._episode_count += 1
    
    @abstractmethod
    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        """Log the current state and optional observation/action/reward.
        
        Called every step during evaluation.
        
        Args:
            state: The full game state (NamedTuple from jaxatari)
            obs: Optional observation (for B2 mode comparison)
            action: Optional action taken
            reward: Optional reward received
        """
        pass
    
    @abstractmethod
    def return_metrics(self) -> Dict[str, float]:
        """Compute and return metrics for the episode.
        
        Called at the end of each episode.
        
        Returns:
            Dictionary mapping metric names to their computed values.
        """
        pass
    
    @abstractmethod
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return descriptions of all metrics for LLM prompt.
        
        Returns:
            Dictionary mapping metric names to human-readable descriptions.
        """
        pass
    
    @property
    def uses_full_state(self) -> bool:
        """Return True if logger has access to full state (B1 setting)."""
        return self.config.info_access == "B1"
    
    @property
    def uses_provided_metrics(self) -> bool:
        """Return True if using provided metrics (A1 setting)."""
        return self.config.metrics_source == "A1"


# =============================================================================
# PONG LOGGER (A1_B1)
# =============================================================================

class PongLogger(Logger):
    """Logger for Pong game with predefined metrics and full state access.
    
    Full State Fields (PongState):
        - player_y: Player paddle Y position
        - player_speed: Player paddle velocity
        - ball_x, ball_y: Ball position
        - ball_vel_x, ball_vel_y: Ball velocity
        - enemy_y, enemy_speed: Enemy paddle state
        - player_score, enemy_score: Current scores
        - step_counter: Frame count
    
    Provided Metrics (A1):
        1. win_rate: Proportion of points won
        2. ball_tracking_error: Average distance from paddle center to ball Y
        3. paddle_efficiency: Movement efficiency (less jitter = better)
        4. avg_rally_length: Average steps between score changes
        5. reaction_time: How quickly paddle responds to ball direction changes
        6. interception_rate: How often paddle is in position when ball arrives
    """
    
    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        super().__init__(ablation_config)
        self._reset_episode_stats()
    
    def _reset_episode_stats(self):
        """Reset per-episode statistics."""
        self._ball_tracking_errors: List[float] = []
        self._paddle_movements: List[float] = []
        self._paddle_positions: List[float] = []
        self._ball_positions_y: List[float] = []
        self._ball_velocities: List[tuple] = []
        self._rally_lengths: List[int] = []
        self._current_rally_length = 0
        self._prev_player_score = 0
        self._prev_enemy_score = 0
        self._prev_paddle_y = None
        self._prev_ball_vel_x = None
        self._direction_changes = 0
        self._reaction_delays: List[int] = []
        self._frames_since_direction_change = 0
        self._interceptions = 0
        self._interception_attempts = 0
        self._points_won = 0
        self._points_lost = 0
        
    def reset(self):
        super().reset()
        self._reset_episode_stats()
    
    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        """Log Pong state for metric computation.
        
        Args:
            state: PongState NamedTuple with full game state, or AtariState containing it
        """
        # Extract inner game state if wrapped
        if hasattr(state, 'env_state'):
            state = state.env_state
        
        # Extract state fields
        player_y = float(state.player_y)
        player_speed = float(state.player_speed)
        ball_x = float(state.ball_x)
        ball_y = float(state.ball_y)
        ball_vel_x = float(state.ball_vel_x)
        ball_vel_y = float(state.ball_vel_y)
        player_score = int(state.player_score)
        enemy_score = int(state.enemy_score)
        
        # Paddle center (assuming height = 16)
        paddle_center = player_y + 8
        
        # 1. Ball tracking error
        tracking_error = abs(ball_y - paddle_center)
        self._ball_tracking_errors.append(tracking_error)
        
        # 2. Paddle movement efficiency
        if self._prev_paddle_y is not None:
            movement = abs(player_y - self._prev_paddle_y)
            self._paddle_movements.append(movement)
        self._prev_paddle_y = player_y
        
        # Store positions for other metrics
        self._paddle_positions.append(paddle_center)
        self._ball_positions_y.append(ball_y)
        self._ball_velocities.append((ball_vel_x, ball_vel_y))
        
        # 3. Rally length tracking
        self._current_rally_length += 1
        if player_score > self._prev_player_score:
            self._rally_lengths.append(self._current_rally_length)
            self._current_rally_length = 0
            self._points_won += 1
        elif enemy_score > self._prev_enemy_score:
            self._rally_lengths.append(self._current_rally_length)
            self._current_rally_length = 0
            self._points_lost += 1
        self._prev_player_score = player_score
        self._prev_enemy_score = enemy_score
        
        # 4. Reaction time tracking (when ball changes direction toward player)
        if self._prev_ball_vel_x is not None:
            # Ball started coming toward player (vel_x > 0 means toward player on right)
            if self._prev_ball_vel_x <= 0 and ball_vel_x > 0:
                self._direction_changes += 1
                self._frames_since_direction_change = 0
        self._prev_ball_vel_x = ball_vel_x
        
        # Track frames since direction change
        if ball_vel_x > 0:  # Ball coming toward player
            self._frames_since_direction_change += 1
            
            # Check if paddle is moving toward ball (reacting)
            if len(self._paddle_movements) >= 2:
                if self._paddle_movements[-1] > 0:  # Paddle moved
                    if self._frames_since_direction_change < 30:  # Quick reaction
                        self._reaction_delays.append(self._frames_since_direction_change)
        
        # 5. Interception tracking
        # When ball is near player's side (x > 120), check if paddle is aligned
        if ball_x > 120 and ball_vel_x > 0:
            self._interception_attempts += 1
            if abs(ball_y - paddle_center) < 12:  # Within paddle height
                self._interceptions += 1
        
        # Store full step data
        self._episode_data.append({
            'step': len(self._episode_data),
            'player_y': player_y,
            'player_speed': player_speed,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_vel_x': ball_vel_x,
            'ball_vel_y': ball_vel_y,
            'player_score': player_score,
            'enemy_score': enemy_score,
            'tracking_error': tracking_error,
            'action': action,
            'reward': reward,
        })
    
    def return_metrics(self) -> Dict[str, float]:
        """Compute and return all Pong metrics."""
        metrics = {}
        
        # 1. Win rate
        total_points = self._points_won + self._points_lost
        metrics['win_rate'] = self._points_won / max(total_points, 1)
        metrics['points_won'] = float(self._points_won)
        metrics['points_lost'] = float(self._points_lost)
        
        # 2. Ball tracking error
        if self._ball_tracking_errors:
            metrics['avg_ball_tracking_error'] = float(np.mean(self._ball_tracking_errors))
            metrics['max_ball_tracking_error'] = float(np.max(self._ball_tracking_errors))
        else:
            metrics['avg_ball_tracking_error'] = 0.0
            metrics['max_ball_tracking_error'] = 0.0
        
        # 3. Paddle efficiency (less movement per unit progress = more efficient)
        if self._paddle_movements:
            total_movement = sum(self._paddle_movements)
            # Efficiency: lower jitter relative to ball tracking improvement
            metrics['total_paddle_movement'] = float(total_movement)
            metrics['avg_paddle_movement'] = float(np.mean(self._paddle_movements))
            
            # Direction changes in paddle (jitter metric)
            direction_changes = sum(1 for i in range(1, len(self._paddle_positions)) 
                                   if (self._paddle_positions[i] - self._paddle_positions[i-1]) *
                                      (self._paddle_positions[i-1] - self._paddle_positions[max(0,i-2)]) < 0)
            metrics['paddle_jitter_count'] = float(direction_changes)
        else:
            metrics['total_paddle_movement'] = 0.0
            metrics['avg_paddle_movement'] = 0.0
            metrics['paddle_jitter_count'] = 0.0
        
        # 4. Rally length
        if self._rally_lengths:
            metrics['avg_rally_length'] = float(np.mean(self._rally_lengths))
            metrics['max_rally_length'] = float(np.max(self._rally_lengths))
        else:
            metrics['avg_rally_length'] = float(self._current_rally_length)
            metrics['max_rally_length'] = float(self._current_rally_length)
        
        # 5. Reaction time
        if self._reaction_delays:
            metrics['avg_reaction_time'] = float(np.mean(self._reaction_delays))
        else:
            metrics['avg_reaction_time'] = 0.0
        
        # 6. Interception rate
        metrics['interception_rate'] = self._interceptions / max(self._interception_attempts, 1)
        
        # Episode length
        metrics['episode_length'] = float(len(self._episode_data))
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return descriptions of all Pong metrics."""
        return {
            'win_rate': 'Proportion of points won by the agent (0.0 to 1.0)',
            'points_won': 'Total points scored by the agent',
            'points_lost': 'Total points lost to the opponent',
            'avg_ball_tracking_error': 'Average pixel distance from paddle center to ball Y position (lower is better)',
            'max_ball_tracking_error': 'Maximum tracking error during episode (lower is better)',
            'total_paddle_movement': 'Total pixels traveled by paddle (efficiency indicator)',
            'avg_paddle_movement': 'Average paddle movement per frame',
            'paddle_jitter_count': 'Number of paddle direction reversals (lower = smoother control)',
            'avg_rally_length': 'Average number of frames per rally (longer = better defense)',
            'max_rally_length': 'Longest rally in the episode',
            'avg_reaction_time': 'Average frames to react when ball comes toward player (lower is better)',
            'interception_rate': 'Proportion of times paddle was positioned to intercept ball (0.0 to 1.0)',
            'episode_length': 'Total frames in the episode',
        }


# =============================================================================
# FREEWAY LOGGER (A1_B1)
# =============================================================================

class FreewayLogger(Logger):
    """Logger for Freeway game with predefined metrics and full state access.
    
    Full State Fields (FreewayState):
        - chicken_y: Chicken Y position (lower = closer to goal)
        - cars: Array of car positions (num_lanes, 2) for x,y
        - score: Current score (successful crossings)
        - time: Game timer
        - cooldown: Cooldown after collision
        - lives_lost: Cumulative lives lost
        - game_over: Game over flag
    
    Provided Metrics (A1):
        1. crossings: Number of successful road crossings
        2. collision_rate: Collisions per attempt
        3. upward_progress: Average Y position improvement per step
        4. lane_transition_rate: How often chicken changes lanes
        5. near_miss_count: Times chicken narrowly avoided a car
        6. time_efficiency: Crossings per unit time
        7. hesitation_rate: Proportion of NOOP actions
    """
    
    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        super().__init__(ablation_config)
        self._reset_episode_stats()
    
    def _reset_episode_stats(self):
        """Reset per-episode statistics."""
        self._chicken_positions: List[float] = []
        self._car_positions_history: List[Any] = []
        self._scores: List[int] = []
        self._reward_crossings = 0.0
        self._cooldowns: List[float] = []
        self._actions: List[int] = []
        self._collisions = 0
        self._near_misses = 0
        self._lane_transitions = 0
        self._prev_chicken_y = None
        self._prev_score = 0
        self._crossing_times: List[int] = []
        self._current_crossing_start = 0
        
    def reset(self):
        super().reset()
        self._reset_episode_stats()
    
    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        """Log Freeway state for metric computation.
        
        Args:
            state: FreewayState NamedTuple with full game state, or AtariState containing it
        """
        # Extract inner game state if wrapped
        if hasattr(state, 'env_state'):
            state = state.env_state
        
        chicken_y = float(state.chicken_y)
        cars = np.array(state.cars)  # Shape: (num_lanes, 2)
        score = int(state.score)
        cooldown = float(state.cooldown)
        
        self._chicken_positions.append(chicken_y)
        self._car_positions_history.append(cars.copy())
        self._scores.append(score)
        self._cooldowns.append(cooldown)
        if action is not None:
            self._actions.append(action)
        if reward is not None and reward > 0:
            self._reward_crossings += float(reward)
        
        # Track lane transitions (significant Y changes)
        if self._prev_chicken_y is not None:
            y_change = chicken_y - self._prev_chicken_y
            if abs(y_change) > 5:  # Significant movement
                self._lane_transitions += 1
        self._prev_chicken_y = chicken_y
        
        # Track collisions (cooldown increase indicates hit)
        if len(self._cooldowns) > 1 and cooldown > self._cooldowns[-2]:
            self._collisions += 1
        
        # Track successful crossings
        if score > self._prev_score:
            frames_to_cross = len(self._episode_data) - self._current_crossing_start
            self._crossing_times.append(frames_to_cross)
            self._current_crossing_start = len(self._episode_data)
        self._prev_score = score
        
        # Track near misses (car very close to chicken but no collision)
        chicken_x = 77  # Chicken is fixed at center X
        for car_pos in cars:
            car_x, car_y = float(car_pos[0]), float(car_pos[1])
            # Near miss: same lane (close Y) and very close X
            if abs(chicken_y - car_y) < 15:  # Same lane
                x_dist = abs(chicken_x - car_x)
                if 8 < x_dist < 20:  # Very close but not collision
                    if cooldown == 0:  # No collision happened
                        self._near_misses += 1
                        break  # Count once per frame
        
        # Store full step data
        self._episode_data.append({
            'step': len(self._episode_data),
            'chicken_y': chicken_y,
            'cars': cars,
            'score': score,
            'cooldown': cooldown,
            'action': action,
            'reward': reward,
        })
    
    def return_metrics(self) -> Dict[str, float]:
        """Compute and return all Freeway metrics."""
        metrics = {}
        
        # 1. Crossings (final score)
        final_score = float(self._scores[-1]) if self._scores else 0.0
        metrics['crossings'] = max(final_score, self._reward_crossings)
        
        # 2. Collision rate
        crossing_attempts = max(metrics['crossings'] + self._collisions, 1)
        metrics['collision_count'] = float(self._collisions)
        metrics['collision_rate'] = self._collisions / crossing_attempts
        
        # 3. Upward progress (average Y change toward goal)
        if len(self._chicken_positions) > 1:
            y_changes = [self._chicken_positions[i-1] - self._chicken_positions[i] 
                        for i in range(1, len(self._chicken_positions))]
            metrics['avg_upward_progress'] = float(np.mean(y_changes))
            metrics['total_upward_progress'] = float(np.sum([max(0, c) for c in y_changes]))
        else:
            metrics['avg_upward_progress'] = 0.0
            metrics['total_upward_progress'] = 0.0
        
        # 4. Lane transition rate
        metrics['lane_transitions'] = float(self._lane_transitions)
        metrics['lane_transition_rate'] = self._lane_transitions / max(len(self._episode_data), 1)
        
        # 5. Near miss count
        metrics['near_miss_count'] = float(self._near_misses)
        
        # 6. Time efficiency (crossings per 1000 frames)
        episode_length = len(self._episode_data)
        metrics['time_efficiency'] = (metrics['crossings'] / max(episode_length, 1)) * 1000
        
        # 7. Average crossing time
        if self._crossing_times:
            metrics['avg_crossing_time'] = float(np.mean(self._crossing_times))
            metrics['min_crossing_time'] = float(np.min(self._crossing_times))
        else:
            metrics['avg_crossing_time'] = float(episode_length)  # Never crossed
            metrics['min_crossing_time'] = float(episode_length)
        
        # 8. Hesitation rate (NOOP actions)
        if self._actions:
            noop_count = sum(1 for a in self._actions if a == 0)
            metrics['hesitation_rate'] = noop_count / len(self._actions)
            metrics['up_action_rate'] = sum(1 for a in self._actions if a == 2) / len(self._actions)
            metrics['down_action_rate'] = sum(1 for a in self._actions if a == 5) / len(self._actions)
        else:
            metrics['hesitation_rate'] = 0.0
            metrics['up_action_rate'] = 0.0
            metrics['down_action_rate'] = 0.0
        
        # Episode length
        metrics['episode_length'] = float(episode_length)
        
        # Position statistics
        if self._chicken_positions:
            metrics['min_y_reached'] = float(np.min(self._chicken_positions))  # Lower = better
            metrics['avg_y_position'] = float(np.mean(self._chicken_positions))
        else:
            metrics['min_y_reached'] = 0.0
            metrics['avg_y_position'] = 0.0
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return descriptions of all Freeway metrics."""
        return {
            'crossings': 'Number of successful road crossings (main score)',
            'collision_count': 'Total number of car collisions',
            'collision_rate': 'Collisions per crossing attempt (lower is better)',
            'avg_upward_progress': 'Average Y movement toward goal per frame (higher is better)',
            'total_upward_progress': 'Total upward movement achieved',
            'lane_transitions': 'Number of lane changes made',
            'lane_transition_rate': 'Lane changes per frame (activity indicator)',
            'near_miss_count': 'Times chicken narrowly avoided collision (risk indicator)',
            'time_efficiency': 'Crossings per 1000 frames (speed indicator)',
            'avg_crossing_time': 'Average frames to complete one crossing',
            'min_crossing_time': 'Fastest crossing in frames',
            'hesitation_rate': 'Proportion of NOOP (no movement) actions',
            'up_action_rate': 'Proportion of UP actions (JAXAtari action id 2)',
            'down_action_rate': 'Proportion of DOWN actions (JAXAtari action id 5)',
            'episode_length': 'Total frames in episode',
            'min_y_reached': 'Lowest Y position reached (lower = closer to goal)',
            'avg_y_position': 'Average chicken Y position during episode',
        }


# =============================================================================
# ASTERIX LOGGER (A1_B1)
# =============================================================================

class AsterixLogger(Logger):
    """Logger for Asterix with lightweight full-state metrics.

    Full State Fields (AsterixState):
        - player_x, player_y: Player position
        - score: Current score
        - lives: Remaining lives
        - enemies / collectibles: Per-lane dynamic entities
        - character_id: 0=Asterix, 1=Obelix
        - stage_cooldown, hit_timer, respawn_timer: Mobility / recovery state

    Provided Metrics (A1):
        1. final_score: Final episode score
        2. lives_remaining: Remaining lives at end
        3. hits_taken: Number of enemy hits
        4. max_stage_reached: Highest lane reached (top lanes are better)
        5. item_pickups: Number of positive reward events
        6. noop_rate: Fraction of NOOP/FIRE actions
        7. diagonal_rate: Fraction of diagonal movement actions
        8. respawn_rate: Fraction of frames spent respawning
    """

    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        super().__init__(ablation_config)
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._scores: List[float] = []
        self._lives: List[int] = []
        self._player_y: List[float] = []
        self._actions: List[int] = []
        self._item_pickups = 0
        self._hits_taken = 0
        self._respawn_frames = 0
        self._collectible_frames = 0
        self._reward_total = 0.0

    def reset(self):
        super().reset()
        self._reset_episode_stats()

    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        if hasattr(state, 'env_state'):
            state = state.env_state

        score = float(state.score)
        lives = int(state.lives)
        player_y = float(state.player_y)
        respawn_timer = int(state.respawn_timer)
        collectible_count = int(np.sum(np.array(state.collectibles.alive)))

        if self._lives and lives < self._lives[-1]:
            self._hits_taken += int(self._lives[-1] - lives)
        if reward is not None and reward > 0:
            self._item_pickups += 1
        if reward is not None:
            self._reward_total += float(reward)
        if respawn_timer > 0:
            self._respawn_frames += 1

        self._scores.append(score)
        self._lives.append(lives)
        self._player_y.append(player_y)
        self._collectible_frames += collectible_count
        if action is not None:
            self._actions.append(int(action))

        self._episode_data.append({
            'step': len(self._episode_data),
            'score': score,
            'lives': lives,
            'player_y': player_y,
            'respawn_timer': respawn_timer,
            'collectible_count': collectible_count,
            'action': action,
            'reward': reward,
        })

    def return_metrics(self) -> Dict[str, float]:
        episode_length = max(len(self._episode_data), 1)
        # Use cumulative reward as the robust episode score because wrapped
        # environments may reset score fields immediately on terminal step.
        final_score = float(self._reward_total)
        lives_remaining = float(self._lives[-1]) if self._lives else 0.0

        metrics = {
            'final_score': final_score,
            'lives_remaining': lives_remaining,
            'hits_taken': float(self._hits_taken),
            'item_pickups': float(self._item_pickups),
            'respawn_rate': self._respawn_frames / episode_length,
            'episode_length': float(episode_length),
            'avg_collectibles_visible': self._collectible_frames / episode_length,
        }

        if self._player_y:
            stage_positions = np.array([23, 39, 55, 71, 87, 103, 119, 135, 151], dtype=float)
            player_y = np.array(self._player_y, dtype=float)
            nearest_stage = np.argmin(np.abs(stage_positions[None, :] - player_y[:, None]), axis=1)
            metrics['max_stage_reached'] = float(np.max(nearest_stage))
            metrics['avg_stage_index'] = float(np.mean(nearest_stage))
            metrics['min_y_reached'] = float(np.min(player_y))
        else:
            metrics['max_stage_reached'] = 0.0
            metrics['avg_stage_index'] = 0.0
            metrics['min_y_reached'] = 0.0

        if self._actions:
            actions = np.array(self._actions, dtype=int)
            metrics['noop_rate'] = float(np.mean(actions == 0))
            metrics['vertical_rate'] = float(np.mean(np.isin(actions, [1, 4])))
            metrics['horizontal_rate'] = float(np.mean(np.isin(actions, [2, 3])))
            metrics['diagonal_rate'] = float(np.mean(np.isin(actions, [5, 6, 7, 8])))
        else:
            metrics['noop_rate'] = 0.0
            metrics['vertical_rate'] = 0.0
            metrics['horizontal_rate'] = 0.0
            metrics['diagonal_rate'] = 0.0

        return metrics

    def get_metric_descriptions(self) -> Dict[str, str]:
        return {
            'final_score': 'Final Asterix score at episode end',
            'lives_remaining': 'Lives remaining at episode end',
            'hits_taken': 'Number of times an enemy collision cost a life',
            'item_pickups': 'Count of positive reward events from collecting items',
            'respawn_rate': 'Fraction of frames spent in respawn recovery',
            'episode_length': 'Total frames in episode',
            'avg_collectibles_visible': 'Average number of visible collectibles on screen',
            'max_stage_reached': 'Highest lane index reached during the episode',
            'avg_stage_index': 'Average lane index occupied during the episode',
            'min_y_reached': 'Lowest player Y reached (higher progress upward)',
            'noop_rate': 'Fraction of NOOP/FIRE actions',
            'vertical_rate': 'Fraction of pure vertical actions',
            'horizontal_rate': 'Fraction of pure horizontal actions',
            'diagonal_rate': 'Fraction of diagonal actions',
        }


# =============================================================================
# BREAKOUT LOGGER (A1_B1)
# =============================================================================

class BreakoutLogger(Logger):
    """Logger for Breakout game with predefined metrics and full state access.
    
    Full State Fields (BreakoutState):
        - player_x: Paddle X position
        - player_speed: Paddle velocity
        - ball_x, ball_y: Ball position
        - ball_vel_x, ball_vel_y: Ball velocity
        - ball_speed_idx: Ball speed level
        - blocks: Block states (which are broken)
        - score: Current score
        - lives: Remaining lives
        - consecutive_paddle_hits: Combo counter
    
    Provided Metrics (A1):
        1. blocks_broken: Total blocks destroyed
        2. ball_tracking_error: Distance from paddle to ball X
        3. paddle_efficiency: Movement relative to ball position
        4. lives_remaining: Lives at end of episode
        5. score: Final score
        6. block_clear_rate: Blocks broken per unit time
        7. hit_streak: Longest consecutive hit streak
    """
    
    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        super().__init__(ablation_config)
        self._reset_episode_stats()
    
    def _reset_episode_stats(self):
        """Reset per-episode statistics."""
        self._ball_tracking_errors: List[float] = []
        self._paddle_movements: List[float] = []
        self._paddle_positions: List[float] = []
        self._ball_positions: List[tuple] = []
        self._block_counts: List[int] = []
        self._scores: List[int] = []
        self._lives_history: List[int] = []
        self._hit_streaks: List[int] = []
        self._current_streak = 0
        self._prev_paddle_x = None
        self._prev_blocks_remaining = 108  # 18 x 6 blocks
        self._prev_lives = 5
        
    def reset(self):
        super().reset()
        self._reset_episode_stats()
    
    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        """Log Breakout state for metric computation.
        
        Args:
            state: BreakoutState NamedTuple with full game state, or AtariState containing it
        """
        # Extract inner game state if wrapped
        if hasattr(state, 'env_state'):
            state = state.env_state
        
        player_x = float(state.player_x)
        player_speed = float(state.player_speed)
        ball_x = float(state.ball_x)
        ball_y = float(state.ball_y)
        ball_vel_x = float(state.ball_vel_x)
        ball_vel_y = float(state.ball_vel_y)
        blocks = np.array(state.blocks)
        score = int(state.score)
        lives = int(state.lives)
        consecutive_hits = int(state.consecutive_paddle_hits)
        
        # Paddle center (assuming width = 16)
        paddle_center = player_x + 8
        
        # 1. Ball tracking error (X distance)
        tracking_error = abs(ball_x - paddle_center)
        self._ball_tracking_errors.append(tracking_error)
        
        # 2. Paddle movement
        if self._prev_paddle_x is not None:
            movement = abs(player_x - self._prev_paddle_x)
            self._paddle_movements.append(movement)
        self._prev_paddle_x = player_x
        
        self._paddle_positions.append(paddle_center)
        self._ball_positions.append((ball_x, ball_y))
        
        # 3. Block count
        blocks_remaining = int(np.sum(blocks))
        self._block_counts.append(blocks_remaining)
        
        # Track block breaks
        if blocks_remaining < self._prev_blocks_remaining:
            self._current_streak += (self._prev_blocks_remaining - blocks_remaining)
        self._prev_blocks_remaining = blocks_remaining
        
        # 4. Score and lives
        self._scores.append(score)
        self._lives_history.append(lives)
        
        # Track life loss (streak reset)
        if lives < self._prev_lives:
            if self._current_streak > 0:
                self._hit_streaks.append(self._current_streak)
            self._current_streak = 0
        self._prev_lives = lives
        
        # Store full step data
        self._episode_data.append({
            'step': len(self._episode_data),
            'player_x': player_x,
            'player_speed': player_speed,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_vel_x': ball_vel_x,
            'ball_vel_y': ball_vel_y,
            'blocks_remaining': blocks_remaining,
            'score': score,
            'lives': lives,
            'consecutive_hits': consecutive_hits,
            'action': action,
            'reward': reward,
        })
    
    def return_metrics(self) -> Dict[str, float]:
        """Compute and return all Breakout metrics."""
        metrics = {}
        
        # 1. Blocks broken
        initial_blocks = 108
        final_blocks = self._block_counts[-1] if self._block_counts else initial_blocks
        metrics['blocks_broken'] = float(initial_blocks - final_blocks)
        metrics['blocks_remaining'] = float(final_blocks)
        metrics['block_clear_percentage'] = (initial_blocks - final_blocks) / initial_blocks
        
        # 2. Ball tracking error
        if self._ball_tracking_errors:
            metrics['avg_ball_tracking_error'] = float(np.mean(self._ball_tracking_errors))
            metrics['max_ball_tracking_error'] = float(np.max(self._ball_tracking_errors))
        else:
            metrics['avg_ball_tracking_error'] = 0.0
            metrics['max_ball_tracking_error'] = 0.0
        
        # 3. Paddle efficiency
        if self._paddle_movements:
            metrics['total_paddle_movement'] = float(sum(self._paddle_movements))
            metrics['avg_paddle_movement'] = float(np.mean(self._paddle_movements))
        else:
            metrics['total_paddle_movement'] = 0.0
            metrics['avg_paddle_movement'] = 0.0
        
        # 4. Lives
        final_lives = self._lives_history[-1] if self._lives_history else 5
        metrics['lives_remaining'] = float(final_lives)
        metrics['lives_lost'] = float(5 - final_lives)
        
        # 5. Score
        metrics['final_score'] = float(self._scores[-1]) if self._scores else 0.0
        
        # 6. Block clear rate (blocks per 1000 frames)
        episode_length = len(self._episode_data)
        metrics['block_clear_rate'] = (metrics['blocks_broken'] / max(episode_length, 1)) * 1000
        
        # 7. Hit streak
        if self._current_streak > 0:
            self._hit_streaks.append(self._current_streak)
        if self._hit_streaks:
            metrics['max_hit_streak'] = float(max(self._hit_streaks))
            metrics['avg_hit_streak'] = float(np.mean(self._hit_streaks))
        else:
            metrics['max_hit_streak'] = 0.0
            metrics['avg_hit_streak'] = 0.0
        
        # Episode length
        metrics['episode_length'] = float(episode_length)
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return descriptions of all Breakout metrics."""
        return {
            'blocks_broken': 'Total number of blocks destroyed',
            'blocks_remaining': 'Blocks still intact at end of episode',
            'block_clear_percentage': 'Proportion of blocks cleared (0.0 to 1.0)',
            'avg_ball_tracking_error': 'Average X distance from paddle center to ball (lower is better)',
            'max_ball_tracking_error': 'Maximum tracking error (lower is better)',
            'total_paddle_movement': 'Total pixels traveled by paddle',
            'avg_paddle_movement': 'Average paddle movement per frame',
            'lives_remaining': 'Lives remaining at end of episode (max 5)',
            'lives_lost': 'Number of lives lost during episode',
            'final_score': 'Final game score',
            'block_clear_rate': 'Blocks broken per 1000 frames (speed indicator)',
            'max_hit_streak': 'Longest streak of blocks broken without losing life',
            'avg_hit_streak': 'Average blocks broken per life',
            'episode_length': 'Total frames in episode',
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_logger(game_name: str, ablation_config: Optional[AblationConfig] = None) -> Logger:
    """Factory function to create a logger for a specific game.
    
    Args:
        game_name: Name of the active thesis game ('pong', 'freeway', 'asterix')
        ablation_config: Internal logger configuration (defaults to provided metrics + full state)
        
    Returns:
        A Logger instance for the specified game
    """
    loggers = {
        'pong': PongLogger,
        'freeway': FreewayLogger,
        'asterix': AsterixLogger,
    }
    
    if game_name.lower() not in loggers:
        available = ', '.join(loggers.keys())
        raise ValueError(f"Unknown game: {game_name}. Available: {available}")
    
    logger_class = loggers[game_name.lower()]
    return logger_class(ablation_config)


def get_all_metric_descriptions(game_name: str) -> Dict[str, str]:
    """Get metric descriptions for a game without creating a full logger.
    
    Useful for including in LLM prompts.
    """
    logger = create_logger(game_name)
    return logger.get_metric_descriptions()


def format_metrics_for_llm(metrics: Dict[str, float], game_or_descriptions) -> str:
    """Format metrics as a readable string for LLM feedback.
    
    Args:
        metrics: Dictionary of metric values
        game_or_descriptions: Either a game name string or a Dict of metric descriptions
        
    Returns:
        Formatted string suitable for LLM prompt
    """
    # Handle both game name and descriptions dict
    if isinstance(game_or_descriptions, str):
        descriptions = get_all_metric_descriptions(game_or_descriptions)
    else:
        descriptions = game_or_descriptions
    
    lines = ["## Episode Metrics\n"]
    
    for name, value in sorted(metrics.items()):
        desc = descriptions.get(name, "No description available")
        if isinstance(value, float):
            if value == int(value):
                value_str = f"{int(value)}"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        lines.append(f"- **{name}**: {value_str}")
        lines.append(f"  - {desc}")
    
    return "\n".join(lines)
