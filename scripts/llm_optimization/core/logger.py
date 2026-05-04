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

    def log_terminal(
        self,
        state: Any,
        obs: Any = None,
        action: int = None,
        reward: float = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a terminal transition when wrappers may have autoreset next_state.

        By default, fall back to logging the provided state. Game-specific loggers
        can override this to avoid reading reset state after SAME_STEP autoreset.
        """
        self.log_state(state, obs=obs, action=action, reward=reward)

    def log_transition(
        self,
        prev_state: Any,
        next_state: Any,
        obs: Any = None,
        action: int = None,
        reward: float = None,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one environment transition.

        The default implementation preserves the existing next-state logging.
        Game-specific loggers can override this when pre-step state is needed
        to interpret terminal/autoreset transitions correctly.
        """
        env_done = bool(info.get("env_done", False)) if isinstance(info, dict) else False
        if env_done:
            self.log_terminal(prev_state, obs=obs, action=action, reward=reward, info=info)
        else:
            self.log_state(next_state, obs=obs, action=action, reward=reward)
    
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

    def _unwrap_state(self, state: Any) -> Any:
        while hasattr(state, 'atari_state'):
            state = state.atari_state
        while hasattr(state, 'env_state'):
            state = state.env_state
        return state
    
    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        """Log Pong state for metric computation.
        
        Args:
            state: PongState NamedTuple with full game state, or AtariState containing it
        """
        # Extract inner game state if wrapped.
        state = self._unwrap_state(state)
        
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

    def _unwrap_state(self, state: Any) -> Any:
        while hasattr(state, 'atari_state'):
            state = state.atari_state
        while hasattr(state, 'env_state'):
            state = state.env_state
        return state
    
    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        """Log Freeway state for metric computation.
        
        Args:
            state: FreewayState NamedTuple with full game state, or AtariState containing it
        """
        # Extract inner game state if wrapped.
        state = self._unwrap_state(state)
        
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
            # Current optimizer-facing Freeway mapping is compact: 0=NOOP, 1=UP, 2=DOWN.
            metrics['up_action_rate'] = sum(1 for a in self._actions if a == 1) / len(self._actions)
            metrics['down_action_rate'] = sum(1 for a in self._actions if a == 2) / len(self._actions)
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
            'up_action_rate': 'Proportion of UP actions in the optimizer-facing mapping (action id 1)',
            'down_action_rate': 'Proportion of DOWN actions in the optimizer-facing mapping (action id 2)',
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
        4. topmost_stage_index: Smallest lane index reached (top lanes are better)
        5. item_pickups: Number of positive reward events
        6. noop_rate: Fraction of NOOP actions
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
        self._terminal_game_over = False
        self._terminal_hit_logged = False
        self._hit_actions: List[int] = []
        self._hit_lanes: List[int] = []
        self._hit_enemy_gaps: List[float] = []
        self._hit_diagonal_count = 0
        self._hit_horizontal_count = 0

    def _unwrap_state(self, state: Any) -> Any:
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        if hasattr(state, 'env_state'):
            state = state.env_state
        return state

    def _lane_index_from_y(self, player_y: float) -> int:
        stage_positions = np.array([23, 39, 55, 71, 87, 103, 119, 135, 151], dtype=float)
        return int(np.argmin(np.abs(stage_positions - player_y)))

    def _record_hit_context(self, state: Any, action: Optional[int]) -> None:
        """Record action/lane context for a life-loss transition."""
        state = self._unwrap_state(state)
        action_int = int(action) if action is not None else -1
        player_y = float(state.player_y)
        player_x = float(state.player_x)
        lane = self._lane_index_from_y(player_y)

        enemies_alive = np.array(state.enemies.alive, dtype=bool)
        enemies_x = np.array(state.enemies.x, dtype=float)
        if 0 <= lane < enemies_x.shape[0] and enemies_alive[lane]:
            enemy_gap = float(abs(enemies_x[lane] - player_x))
        elif np.any(enemies_alive):
            enemy_gap = float(np.min(np.abs(enemies_x[enemies_alive] - player_x)))
        else:
            enemy_gap = float("nan")

        self._hit_actions.append(action_int)
        self._hit_lanes.append(lane)
        self._hit_enemy_gaps.append(enemy_gap)
        if action_int in [4, 5, 6, 7, 8]:
            self._hit_diagonal_count += 1
        if action_int in [1, 2]:
            self._hit_horizontal_count += 1

    def reset(self):
        super().reset()
        self._reset_episode_stats()

    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        state = self._unwrap_state(state)

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

    def log_transition(
        self,
        prev_state: Any,
        next_state: Any,
        obs: Any = None,
        action: int = None,
        reward: float = None,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        prev_game_state = self._unwrap_state(prev_state)
        next_game_state = self._unwrap_state(next_state)
        env_done = bool(info.get("env_done", False)) if isinstance(info, dict) else False

        if env_done:
            self.log_terminal(prev_game_state, obs=obs, action=action, reward=reward, info=info)
            return

        prev_lives = int(prev_game_state.lives)
        next_lives = int(next_game_state.lives)
        if next_lives < prev_lives:
            self._record_hit_context(prev_game_state, action)

        self.log_state(next_game_state, obs=obs, action=action, reward=reward)

    def log_terminal(
        self,
        state: Any,
        obs: Any = None,
        action: int = None,
        reward: float = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log terminal Asterix step from the pre-step state.

        ObjectCentricWrapper autoresets after env_done, so the next_state returned
        to Python can already be a fresh reset state. For terminal game-over steps,
        use the pre-step state and explicitly record the final life loss.
        """
        state = self._unwrap_state(state)

        score = float(state.score)
        previous_lives = int(state.lives)
        player_y = float(state.player_y)
        collectible_count = int(np.sum(np.array(state.collectibles.alive)))

        if reward is not None and reward > 0:
            self._item_pickups += 1
        if reward is not None:
            self._reward_total += float(reward)

        # Asterix game_over is caused by the final enemy collision. Non-terminal
        # life losses are observed normally in log_state; the terminal one would
        # otherwise be hidden by wrapper autoreset.
        if not self._terminal_hit_logged:
            self._hits_taken += 1
            self._terminal_hit_logged = True
            self._record_hit_context(state, action)

        self._terminal_game_over = True
        self._scores.append(score)
        self._lives.append(0)
        self._player_y.append(player_y)
        self._collectible_frames += collectible_count
        if action is not None:
            self._actions.append(int(action))

        self._episode_data.append({
            'step': len(self._episode_data),
            'score': score,
            'lives': 0,
            'previous_lives': previous_lives,
            'player_y': player_y,
            'respawn_timer': 0,
            'collectible_count': collectible_count,
            'action': action,
            'reward': reward,
            'terminal_game_over': True,
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
            'terminal_game_over': float(self._terminal_game_over),
            'item_pickups': float(self._item_pickups),
            'item_pickup_rate_per_1000': (self._item_pickups / episode_length) * 1000.0,
            'score_rate_per_1000': (final_score / episode_length) * 1000.0,
            'hit_rate_per_1000': (self._hits_taken / episode_length) * 1000.0,
            'hit_diagonal_fraction': self._hit_diagonal_count / max(self._hits_taken, 1),
            'hit_horizontal_fraction': self._hit_horizontal_count / max(self._hits_taken, 1),
            'respawn_rate': self._respawn_frames / episode_length,
            'episode_length': float(episode_length),
            'avg_collectibles_visible': self._collectible_frames / episode_length,
        }

        if self._hit_lanes:
            metrics['avg_hit_lane_index'] = float(np.mean(self._hit_lanes))
            finite_gaps = [gap for gap in self._hit_enemy_gaps if np.isfinite(gap)]
            metrics['avg_hit_enemy_gap'] = float(np.mean(finite_gaps)) if finite_gaps else float("nan")
        else:
            metrics['avg_hit_lane_index'] = 0.0
            metrics['avg_hit_enemy_gap'] = float("nan")

        if self._player_y:
            stage_positions = np.array([23, 39, 55, 71, 87, 103, 119, 135, 151], dtype=float)
            player_y = np.array(self._player_y, dtype=float)
            nearest_stage = np.argmin(np.abs(stage_positions[None, :] - player_y[:, None]), axis=1)
            metrics['topmost_stage_index'] = float(np.min(nearest_stage))
            metrics['bottommost_stage_index'] = float(np.max(nearest_stage))
            metrics['max_stage_reached'] = float(np.max(nearest_stage))
            metrics['avg_stage_index'] = float(np.mean(nearest_stage))
            metrics['min_y_reached'] = float(np.min(player_y))
        else:
            metrics['topmost_stage_index'] = 0.0
            metrics['bottommost_stage_index'] = 0.0
            metrics['max_stage_reached'] = 0.0
            metrics['avg_stage_index'] = 0.0
            metrics['min_y_reached'] = 0.0

        if self._actions:
            actions = np.array(self._actions, dtype=int)
            metrics['noop_rate'] = float(np.mean(actions == 0))
            metrics['vertical_rate'] = float(np.mean(actions == 3))
            metrics['horizontal_rate'] = float(np.mean(np.isin(actions, [1, 2])))
            metrics['diagonal_rate'] = float(np.mean(np.isin(actions, [4, 5, 6, 7, 8])))
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
            'terminal_game_over': '1 if the episode ended by game over, else 0',
            'item_pickups': 'Count of positive reward events from collecting items',
            'item_pickup_rate_per_1000': 'Item pickups per 1000 frames',
            'score_rate_per_1000': 'Score points per 1000 frames',
            'hit_rate_per_1000': 'Life-losing hits per 1000 frames',
            'hit_diagonal_fraction': 'Fraction of life-losing hits where the chosen action was diagonal',
            'hit_horizontal_fraction': 'Fraction of life-losing hits where the chosen action was pure horizontal',
            'avg_hit_lane_index': 'Average lane index where life-losing hits occurred',
            'avg_hit_enemy_gap': 'Average absolute x-gap to the relevant enemy before life-losing hits',
            'respawn_rate': 'Fraction of frames spent in respawn recovery',
            'episode_length': 'Total frames in episode',
            'avg_collectibles_visible': 'Average number of visible collectibles on screen',
            'topmost_stage_index': 'Smallest lane index reached; 0 is top lane, 7 is bottom lane',
            'bottommost_stage_index': 'Largest lane index reached; 7 is bottom lane',
            'max_stage_reached': 'Backward-compatible alias for bottommost_stage_index',
            'avg_stage_index': 'Average lane index occupied; lower means more time in upper lanes',
            'min_y_reached': 'Lowest player Y reached (higher progress upward)',
            'noop_rate': 'Fraction of NOOP actions',
            'vertical_rate': 'Fraction of pure DOWN actions in the wrapped optimizer-facing mapping (action=3)',
            'horizontal_rate': 'Fraction of pure horizontal actions in the wrapped mapping (RIGHT=1 or LEFT=2)',
            'diagonal_rate': 'Fraction of diagonal actions in the wrapped mapping (4-8)',
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
        # Extract inner game state if wrapped. The current unified path passes
        # ObjectCentricState -> AtariState -> BreakoutState.
        if hasattr(state, 'atari_state'):
            state = state.atari_state
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
# SKIING LOGGER (A1_B1)
# =============================================================================

class SkiingLogger(Logger):
    """Logger for Skiing with full-state metrics."""

    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        super().__init__(ablation_config)
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._gates_passed: List[int] = []
        self._gates_seen: List[int] = []
        self._remaining_gates: List[int] = []
        self._speed_y: List[float] = []
        self._gate_center_errors: List[float] = []
        self._collision_count = 0
        self._tree_collisions = 0
        self._mogul_collisions = 0
        self._flag_collisions = 0
        self._previous_fell = 0

    def reset(self):
        super().reset()
        self._reset_episode_stats()

    def _unwrap_state(self, state: Any) -> Any:
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        if hasattr(state, 'env_state'):
            state = state.env_state
        return state

    def _next_gate_error(self, state: Any) -> float:
        flags = np.array(state.flags, dtype=float)
        skier_x = float(state.skier_x)
        skier_y = 46.0
        visible = flags[:, 1] < 210.0
        ahead = np.logical_and(visible, flags[:, 1] >= skier_y - 8.0)
        if np.any(ahead):
            candidates = np.where(ahead)[0]
            idx = int(candidates[np.argmin(flags[candidates, 1])])
        elif np.any(visible):
            candidates = np.where(visible)[0]
            idx = int(candidates[np.argmin(np.abs(flags[candidates, 1] - skier_y))])
        else:
            return float("nan")
        gate_center = float(flags[idx, 0] + 16.0)
        return abs(skier_x - gate_center)

    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        state = self._unwrap_state(state)

        remaining = int(state.successful_gates)
        gates_passed = int(20 - remaining)
        gates_seen = int(state.gates_seen)
        skier_fell = int(state.skier_fell)
        collision_type = int(state.collision_type)

        if self._previous_fell == 0 and skier_fell > 0:
            self._collision_count += 1
            if collision_type == 1:
                self._tree_collisions += 1
            elif collision_type == 2:
                self._mogul_collisions += 1
            elif collision_type == 3:
                self._flag_collisions += 1
        self._previous_fell = skier_fell

        self._gates_passed.append(gates_passed)
        self._gates_seen.append(gates_seen)
        self._remaining_gates.append(remaining)
        self._speed_y.append(float(state.skier_y_speed))
        self._gate_center_errors.append(self._next_gate_error(state))

        if action is not None:
            self._actions.append(int(action))
        if reward is not None:
            self._rewards.append(float(reward))

        self._episode_data.append({
            'step': len(self._episode_data),
            'gates_passed': gates_passed,
            'gates_seen': gates_seen,
            'remaining_gates': remaining,
            'skier_x': float(state.skier_x),
            'skier_pos': int(state.skier_pos),
            'skier_fell': skier_fell,
            'collision_type': collision_type,
            'skier_y_speed': float(state.skier_y_speed),
            'action': action,
            'reward': reward,
        })

    def return_metrics(self) -> Dict[str, float]:
        actions = np.asarray(self._actions, dtype=int) if self._actions else np.asarray([], dtype=int)
        final_remaining = float(self._remaining_gates[-1]) if self._remaining_gates else 20.0
        final_seen = float(self._gates_seen[-1]) if self._gates_seen else 0.0
        final_passed = float(self._gates_passed[-1]) if self._gates_passed else 0.0
        gate_errors = np.asarray(self._gate_center_errors, dtype=float)
        finite_gate_errors = gate_errors[np.isfinite(gate_errors)]

        return {
            'gates_passed': final_passed,
            'missed_gates': final_remaining,
            'gates_seen': final_seen,
            'gate_pass_rate': final_passed / max(final_seen, final_passed, 1.0),
            'collision_count': float(self._collision_count),
            'tree_collisions': float(self._tree_collisions),
            'mogul_collisions': float(self._mogul_collisions),
            'flag_collisions': float(self._flag_collisions),
            'average_gate_center_error': float(np.mean(finite_gate_errors)) if finite_gate_errors.size else float("nan"),
            'average_speed_y': float(np.mean(self._speed_y)) if self._speed_y else 0.0,
            'reward_total': float(np.sum(self._rewards)) if self._rewards else 0.0,
            'noop_rate': float(np.mean(actions == 0)) if actions.size else 0.0,
            'right_action_rate': float(np.mean(actions == 1)) if actions.size else 0.0,
            'left_action_rate': float(np.mean(actions == 2)) if actions.size else 0.0,
            'fire_action_rate': float(np.mean(actions == 3)) if actions.size else 0.0,
            'down_action_rate': float(np.mean(actions == 4)) if actions.size else 0.0,
            'turn_rate': float(np.mean(np.isin(actions, [1, 2]))) if actions.size else 0.0,
            'average_steps': float(len(self._episode_data)),
        }

    def get_metric_descriptions(self) -> Dict[str, str]:
        return {
            'gates_passed': 'Number of gates successfully passed. Higher is better; maximum is 20.',
            'missed_gates': 'Remaining missed-gate counter at episode end. Lower is better.',
            'gates_seen': 'Number of gates processed by the environment.',
            'gate_pass_rate': 'Fraction of processed gates that were passed successfully.',
            'collision_count': 'Number of recovery-triggering collisions.',
            'tree_collisions': 'Number of collisions with trees.',
            'mogul_collisions': 'Number of collisions with moguls.',
            'flag_collisions': 'Number of collisions with flag poles.',
            'average_gate_center_error': 'Average absolute distance from skier_x to the next gate center.',
            'average_speed_y': 'Average downhill speed; higher usually finishes faster if gate alignment is safe.',
            'reward_total': 'Total logged negative ALE-style reward; less negative is better.',
            'noop_rate': 'Fraction of NOOP actions.',
            'right_action_rate': 'Fraction of RIGHT steering actions.',
            'left_action_rate': 'Fraction of LEFT steering actions.',
            'fire_action_rate': 'Fraction of FIRE/jump actions.',
            'down_action_rate': 'Fraction of DOWN/tuck actions.',
            'turn_rate': 'Fraction of LEFT or RIGHT actions.',
            'average_steps': 'Number of logged steps before episode end or evaluation truncation.',
        }


# =============================================================================
# KANGAROO LOGGER (A1_B1)
# =============================================================================

class KangarooLogger(Logger):
    """Logger for Kangaroo exploratory onboarding.

    The initial Kangaroo smoke run produced a valid but zero-score policy, so
    these metrics focus on basic behavior diagnosis: action diversity, vertical
    progress, ladder/climbing behavior, score events, and life loss.
    """

    def __init__(self, ablation_config: Optional[AblationConfig] = None):
        super().__init__(ablation_config)
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._score_values: List[float] = []
        self._lives_values: List[int] = []
        self._player_x: List[float] = []
        self._player_y: List[float] = []
        self._is_climbing: List[float] = []
        self._is_crashing: List[float] = []

    def reset(self):
        super().reset()
        self._reset_episode_stats()

    def _unwrap_state(self, state: Any) -> Any:
        while True:
            if hasattr(state, "atari_state"):
                state = state.atari_state
            elif hasattr(state, "env_state"):
                state = state.env_state
            else:
                return state

    def log_state(self, state: Any, obs: Any = None, action: int = None, reward: float = None) -> None:
        state = self._unwrap_state(state)
        player = state.player

        self._actions.append(int(action) if action is not None else -1)
        self._rewards.append(float(reward) if reward is not None else 0.0)
        self._score_values.append(float(state.score))
        self._lives_values.append(int(state.lives))
        self._player_x.append(float(player.x))
        self._player_y.append(float(player.y))
        self._is_climbing.append(float(player.is_climbing))
        self._is_crashing.append(float(player.is_crashing))

    def return_metrics(self) -> Dict[str, float]:
        if not self._actions:
            return {
                "average_steps": 0.0,
                "reward_total": 0.0,
                "score_events": 0.0,
            }

        actions = np.asarray(self._actions, dtype=int)
        rewards = np.asarray(self._rewards, dtype=float)
        xs = np.asarray(self._player_x, dtype=float)
        ys = np.asarray(self._player_y, dtype=float)
        lives = np.asarray(self._lives_values, dtype=float)

        horizontal_mask = np.isin(actions, [3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17])
        fire_mask = np.isin(actions, [1, 10, 11, 12, 13, 14, 15, 16, 17])
        punch_mask = np.isin(actions, [11, 12])
        up_mask = np.isin(actions, [2, 6, 7, 10, 14, 15])
        down_mask = np.isin(actions, [5, 8, 9, 13, 16, 17])
        noop_mask = actions == 0

        start_y = ys[0]
        min_y = np.min(ys)
        final_y = ys[-1]
        start_lives = lives[0] if lives.size else 0.0
        final_lives = lives[-1] if lives.size else 0.0
        score_events = float(np.sum(rewards > 0))
        up_action_count = float(np.sum(up_mask))
        max_upward_progress = float(start_y - min_y)
        net_upward_progress = float(start_y - final_y)
        x_range = float(np.max(xs) - np.min(xs)) if xs.size else 0.0
        y_range = float(np.max(ys) - np.min(ys)) if ys.size else 0.0
        up_action_rate = float(np.mean(up_mask))
        horizontal_action_rate = float(np.mean(horizontal_mask))
        fire_action_rate = float(np.mean(fire_mask))
        punch_action_rate = float(np.mean(punch_mask))
        climb_rate = float(np.mean(self._is_climbing)) if self._is_climbing else 0.0

        return {
            "average_steps": float(len(actions)),
            "reward_total": float(np.sum(rewards)),
            "score_events": score_events,
            "final_score": float(self._score_values[-1]) if self._score_values else 0.0,
            "lives_lost": float(max(0.0, start_lives - final_lives)),
            "final_lives": float(final_lives),
            "min_player_y": float(min_y),
            "final_player_y": float(final_y),
            "max_upward_progress": max_upward_progress,
            "net_upward_progress": net_upward_progress,
            "x_range": x_range,
            "y_range": y_range,
            "up_action_rate": up_action_rate,
            "horizontal_action_rate": horizontal_action_rate,
            "fire_action_rate": fire_action_rate,
            "punch_action_rate": punch_action_rate,
            "rightfire_action_rate": float(np.mean(actions == 11)),
            "leftfire_action_rate": float(np.mean(actions == 12)),
            "down_action_rate": float(np.mean(down_mask)),
            "noop_rate": float(np.mean(noop_mask)),
            "unique_action_count": float(len(set(self._actions))),
            "climb_rate": climb_rate,
            "crash_rate": float(np.mean(self._is_crashing)) if self._is_crashing else 0.0,
            "up_only_failure": float(up_action_rate > 0.95 and len(set(self._actions)) <= 1),
            "horizontal_only_failure": float(horizontal_action_rate > 0.95 and max_upward_progress < 5.0),
            "score_without_climb_pattern": float(score_events > 0 and up_action_count == 0.0 and max_upward_progress < 10.0),
            "punch_farming_pattern": float(score_events >= 2.0 and punch_action_rate > 0.05 and up_action_count == 0.0),
            "first_reward_suppressed_pattern": float(
                score_events == 0.0
                and fire_action_rate < 0.01
                and x_range > 40.0
                and max_upward_progress >= 8.0
                and climb_rate > 0.05
            ),
            "no_fire_scoreless_pattern": float(
                score_events == 0.0
                and fire_action_rate < 0.01
                and x_range > 20.0
            ),
        }

    def get_metric_descriptions(self) -> Dict[str, str]:
        return {
            "average_steps": "Number of logged steps before episode end or evaluation truncation.",
            "reward_total": "Total reward during the logged rollout.",
            "score_events": "Number of positive reward events, such as fruit, enemies, or level progress.",
            "final_score": "Raw game score at the end of the logged rollout.",
            "lives_lost": "Number of lives lost during the logged rollout.",
            "final_lives": "Lives remaining at the end of the logged rollout.",
            "min_player_y": "Highest screen position reached by the player; lower y means higher on screen.",
            "final_player_y": "Player y position at the end of the logged rollout.",
            "max_upward_progress": "Largest upward displacement from the starting y position.",
            "net_upward_progress": "Final upward displacement from the starting y position.",
            "x_range": "Horizontal movement range; near zero means the policy barely moved left or right.",
            "y_range": "Vertical movement range during the rollout.",
            "up_action_rate": "Fraction of actions that include upward movement.",
            "horizontal_action_rate": "Fraction of actions with left or right movement.",
            "fire_action_rate": "Fraction of actions using FIRE.",
            "punch_action_rate": "Fraction of actions using LEFTFIRE or RIGHTFIRE.",
            "rightfire_action_rate": "Fraction of actions using RIGHTFIRE.",
            "leftfire_action_rate": "Fraction of actions using LEFTFIRE.",
            "down_action_rate": "Fraction of actions with downward movement.",
            "noop_rate": "Fraction of NOOP actions.",
            "unique_action_count": "Number of distinct actions used.",
            "climb_rate": "Fraction of logged states where the player is in the climbing state.",
            "crash_rate": "Fraction of logged states where the player is crashing.",
            "up_only_failure": "1 when the policy effectively returns only UP, a known failed Kangaroo smoke-run pattern.",
            "horizontal_only_failure": "1 when the policy mostly moves left/right but makes almost no upward progress.",
            "score_without_climb_pattern": "1 when the policy scores without any UP action or meaningful upward progress.",
            "punch_farming_pattern": "1 when multiple score events come from a high punch-action rate without any UP action.",
            "first_reward_suppressed_pattern": "1 when a scoreless policy moves/climbs through the first route but almost never uses FIRE, often meaning it suppressed the working first RIGHTFIRE reward.",
            "no_fire_scoreless_pattern": "1 when a scoreless policy moves around but almost never uses FIRE, indicating the punch/jump branch may have been removed or over-gated.",
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
        'breakout': BreakoutLogger,
        'skiing': SkiingLogger,
        'kangaroo': KangarooLogger,
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
