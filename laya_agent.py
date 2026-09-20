"""Laya-backed controller for DangerousDaveEnv.

Laya (Convai Innovations) is, like Jev, a text/JSON-in, typed-judgment-out model - not a
vision model and not built for arithmetic or multi-hop spatial reasoning. So this module does
the geometry itself (reading the level's tile grid and labelling what's near Dave in relative,
human-readable terms) and only asks Laya to make the actual judgment call: which of the 7
discrete actions to take next.

Unlike Jev, Laya runs entirely locally (no API key): laya.load("convaiinnovations/laya") loads
the English checkpoint (ModernBERT-large) - Single Model Mode, since the game state here is
always English text and Router mode's language detection would be wasted work. The first call
downloads the ~421M-parameter checkpoint from Hugging Face Hub (one-time, cached thereafter).
"""

import logging

import laya

from ddave.utils import DIRECTION, WIDTH_OF_MAP_NODE, HEIGHT_OF_MAP_NODE
from env import LOCKED_DOOR

logger = logging.getLogger("laya_agent")

# Matches the exact action-int mapping used by DangerousDaveEnv.step() (env.py:141-148)
ACTION_NAMES = {
    0: "up",
    1: "left",
    2: "right",
    3: "down",
    4: "up_left",
    5: "up_right",
    6: "noop",
}
NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}

ACTION_CRITERIA = {
    "up": "Move up: climbs if on a tree, otherwise jumps straight up.",
    "left": "Walk left along the ground or in the air.",
    "right": "Walk right along the ground or in the air.",
    "down": "Duck if standing still, or climb down if on a tree.",
    "up_left": "Jump or climb while moving left.",
    "up_right": "Jump or climb while moving right.",
    "noop": "Do nothing this decision: keep falling under gravity or stay put.",
}

# Tile ids (Tile.getId()) that are dangerous to touch, per InteractiveScenery in ddave/utils.py.
HAZARD_TILE_IDS = {"fire", "tentacles", "water"}
# Tile ids that block movement (Solid subclasses in ddave/utils.py) - despite the name,
# "tunnel" and "pinkpipe" are solid walls in this recreation, not passable shortcuts.
SOLID_TILE_IDS = {"solid", "pinkpipe", "tunnel"}

# Laya only knows what a tile *is* from these engine-internal id strings (e.g. "tunnel" reads
# as an inviting passage, but here it's a wall) - so every nearby tile is labelled with what it
# actually does to Dave, not left for Laya to guess from the name.
TILE_CATEGORY = {
    **{tile_id: "wall, blocks movement" for tile_id in SOLID_TILE_IDS},
    **{tile_id: "deadly hazard, avoid touching" for tile_id in HAZARD_TILE_IDS},
    "items": "collectible, safe to walk into",
    "trophy": "collectible, safe to walk into, needed to open a locked door",
    "gun": "collectible, safe to walk into",
    "jetpack": "collectible, safe to walk into",
    "tree": "climbable, safe, lets Dave climb up/down",
    "door": "level exit, safe to walk into",
    "player_spawner": "safe ground",
}


def _classify_tile(tile_id):
    return TILE_CATEGORY.get(tile_id, "open space, safe to move through")

# Local window (in tiles) scanned around the player each decision, biased ahead of Dave
# rather than covering the whole (up to 150-tile-wide) level - irrelevant context measurably
# hurts these models, per TypeSafe's own docs (and the same reasoning applies to Laya here).
WINDOW_BEHIND = 1
WINDOW_AHEAD = 6
WINDOW_UP = 2
WINDOW_DOWN = 2

DEFAULT_CONFIDENCE_THRESHOLD = 0.35


def _offset_words(dx, dy):
    parts = []
    if dy < 0:
        parts.append(f"{-dy} up")
    elif dy > 0:
        parts.append(f"{dy} down")
    if dx < 0:
        parts.append(f"{-dx} left")
    elif dx > 0:
        parts.append(f"{dx} right")
    return ", ".join(parts) if parts else "here"


class LayaAgent:
    """Drop-in controller for DangerousDaveEnv: call .act(env) to get the next action int."""

    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        # Single Model Mode, English checkpoint only - the game state here is always English
        # text, so there's nothing for Router mode's language detection to do.
        self.model = laya.load("convaiinnovations/laya")
        self.confidence_threshold = confidence_threshold
        self._previous_action = 6  # noop until Laya makes its first call
        # Latest decision, kept for the on-screen HUD (see play_laya.py):
        # {"action_name": str, "confidence": float | None, "error": str | None}
        self.last_decision = None

    def build_state(self, env):
        level = env.Level
        player = env.GamePlayer

        tile_x = round(env.player_position_x / WIDTH_OF_MAP_NODE)
        tile_y = round(env.player_position_y / HEIGHT_OF_MAP_NODE)
        facing_right = player.direction_x != DIRECTION.LEFT

        if facing_right:
            x_min, x_max = -WINDOW_BEHIND, WINDOW_AHEAD
        else:
            x_min, x_max = -WINDOW_AHEAD, WINDOW_BEHIND
        x_range = range(x_min, x_max + 1)

        def tile_at(dx, dy):
            x, y = tile_x + dx, tile_y + dy
            if not level.validateCoordinates(x, y):
                return None
            return level.getNode(x, y).getId()

        nearby_tiles = []
        for dx in x_range:
            for dy in range(-WINDOW_UP, WINDOW_DOWN + 1):
                if dx == 0 and dy == 0:
                    continue
                tile_id = tile_at(dx, dy)
                if tile_id in (None, "scenery", "undefined"):
                    continue  # out of bounds or empty background - not worth Laya's attention
                nearby_tiles.append(
                    {"offset": _offset_words(dx, dy), "tile": tile_id, "what_it_does": _classify_tile(tile_id)}
                )

        # Coarse heuristics computed in code (not left for Laya to infer from raw geometry):
        step = 1 if facing_right else -1
        ahead_head_tile = tile_at(step, 0)
        ahead_ground_tile = tile_at(step, 1)
        above_tile = tile_at(0, -1)
        # "Gap" is approximate: true when the tile at foot-level one step ahead isn't solid
        # ground (open air/scenery). Laya still sees the full nearby_tiles list to cross-check.
        gap_ahead = ahead_ground_tile is not None and ahead_ground_tile not in SOLID_TILE_IDS

        return {
            "player": {
                "facing": "right" if facing_right else "left",
                "state": player.getCurrentState().name,
                "lives": player.lives,
                "score": player.score,
                "has_jetpack": bool(player.inventory["jetpack"]),
                "has_gun": bool(player.inventory["gun"]),
                "has_trophy": bool(player.inventory["trophy"]),
                "on_or_near_tree": bool(player.inventory["tree"]),
            },
            "nearby_tiles": nearby_tiles,
            "solid_directly_ahead": ahead_head_tile in SOLID_TILE_IDS,
            "hazard_directly_ahead": ahead_head_tile in HAZARD_TILE_IDS or ahead_ground_tile in HAZARD_TILE_IDS,
            "gap_directly_ahead": gap_ahead,
            "solid_directly_above": above_tile in SOLID_TILE_IDS,
            "level_needs_trophy_for_door": bool(LOCKED_DOOR),
        }

    def act(self, env):
        state = self.build_state(env)
        try:
            result = self.model.predict(
                state,
                {
                    "action": {
                        "type": "choice",
                        "instructions": (
                            "You are playing Dangerous Dave, a side-scrolling platformer, trying "
                            "to make progress through the level. Given Dave's current situation, "
                            "which action should he take next? Each tile in `nearby_tiles` has a "
                            "`what_it_does` field telling you exactly what happens if Dave touches "
                            "it - trust that field over what the raw tile name suggests (for "
                            "example a tile named 'tunnel' is actually a solid wall here, not a "
                            "passage). Never move toward a tile whose `what_it_does` says it "
                            "blocks movement or is a deadly hazard; prefer moving toward open "
                            "space, collectibles, or the level exit instead."
                        ),
                        "criteria": ACTION_CRITERIA,
                    },
                },
            )
        except Exception as error:
            # laya has no dedicated error type for a bad/failed local inference call (unlike
            # TypeSafeError for Jev's hosted API) - catch broadly so a transient issue costs
            # Dave a life rather than crashing the whole run.
            logger.warning("Laya call failed (%s); repeating previous action", error)
            self.last_decision = {
                "action_name": ACTION_NAMES[self._previous_action],
                "confidence": None,
                "error": str(error),
            }
            return self._previous_action

        answer = result["answers"]["action"]
        action = NAME_TO_ACTION.get(answer["choice"])
        if action is None:
            logger.warning("Unrecognized choice %r; repeating previous action", answer["choice"])
            action = self._previous_action

        # Same reasoning as Jev: with 7 roughly-plausible options, confidence commonly sits well
        # below what a binary/few-option question would produce even for a reasonable pick. A
        # wrong move here just costs a life and respawns Dave (cheap to recover from), so we
        # still act on the top choice and only use confidence to flag it as uncertain for the
        # HUD, rather than freezing gameplay by falling back to the previous action.
        confidence = answer["confidence"]
        low_confidence = confidence < self.confidence_threshold
        self.last_decision = {
            "action_name": ACTION_NAMES[action],
            "confidence": confidence,
            "error": None,
            "low_confidence": low_confidence,
        }
        print(f"Laya action: {ACTION_NAMES[action]} (confidence={confidence:.2f})")
        self._previous_action = action
        return action
