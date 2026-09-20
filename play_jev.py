"""Watch Jev (TypeSafe AI) play Dangerous Dave.

Usage:
    python play_jev.py

Requires TYPESAFE_API_KEY to be set (e.g. in a .env file - see .env.example).
"""

import pygame
from dotenv import load_dotenv

load_dotenv()

# The game (ddave/utils.py: Screen) always creates its window at a small fixed resolution via
# pygame.display.set_mode(). Rather than touch Screen - which every entry point (game.py,
# env.py, agent.py) also relies on - intercept that one call so it opens fullscreen and lets
# pygame upscale (with letterboxing, so pixel art isn't stretched/distorted) instead. Screen
# still receives back a surface at the same small logical size it asked for and draws onto it
# exactly as before; only this script is affected, since the patch is applied here, not there.
_real_set_mode = pygame.display.set_mode


def _fullscreen_set_mode(size, *args, **kwargs):
    try:
        return _real_set_mode(size, pygame.SCALED | pygame.FULLSCREEN)
    except pygame.error:
        # Some environments (remote desktops, unusual drivers) reject exclusive fullscreen -
        # fall back to a large resizable window rather than crash the whole script.
        return _real_set_mode(size, pygame.SCALED | pygame.RESIZABLE)


pygame.display.set_mode = _fullscreen_set_mode

from ddave.utils import TILE_SCALE_FACTOR, TOP_OVERLAY_POS  # noqa: E402
from env import DangerousDaveEnv  # noqa: E402
from jev_agent import JevAgent  # noqa: E402

HUD_COLOR_NORMAL = (255, 255, 255)
HUD_COLOR_UNCERTAIN = (255, 210, 60)
HUD_COLOR_ERROR = (255, 90, 90)
HUD_BORDER_COLOR = (255, 255, 255)
HUD_HIGHLIGHT_COLOR = (50, 140, 80)  # background behind the row for the action actually taken
HUD_FONT_SIZE = 32
HUD_OPTION_FONT_SIZE = 22
HUD_MARGIN = 10
HUD_PADDING = (20, 14)
HUD_ROW_GAP = 3
# Sits just below the game's own top overlay bar (score/level/lives), in the plain gameplay
# area, so it doesn't cover any existing UI.
HUD_Y = TOP_OVERLAY_POS * TILE_SCALE_FACTOR + HUD_MARGIN


def draw_jev_hud(header_font, option_font, decision):
    """Draws a badge showing Jev's chosen action plus its confidence across *all* 7 options
    each decision (not just the one taken), with the taken option's row highlighted, so the
    full distribution behind the pick is visible directly on the game window."""
    rows = []  # (option_name, probability) pairs for every action, in ACTION_NAMES order
    if decision is None:
        header_text, header_color = "JEV: THINKING...", HUD_COLOR_NORMAL
    elif decision["error"]:
        header_text = f"JEV: {decision['action_name'].upper()} - API ERROR, REUSING LAST MOVE"
        header_color = HUD_COLOR_ERROR
    else:
        suffix = "  UNSURE" if decision["low_confidence"] else ""
        header_text = f"JEV: {decision['action_name'].upper()}  ({decision['confidence'] * 100:.0f}% CONFIDENCE{suffix})"
        header_color = HUD_COLOR_UNCERTAIN if decision["low_confidence"] else HUD_COLOR_NORMAL
        rows = list(decision["probabilities"].items())

    header_surface = header_font.render(header_text, True, header_color)
    row_data = [
        (option_font.render(f"{'> ' if name == decision['action_name'] else '  '}{name.upper():<8}{prob * 100:5.1f}%",
                             True, HUD_COLOR_NORMAL),
         name == decision["action_name"])
        for name, prob in rows
    ]

    content_width = max([header_surface.get_width()] + [surface.get_width() for surface, _ in row_data])
    content_height = header_surface.get_height()
    if row_data:
        content_height += HUD_ROW_GAP + sum(s.get_height() for s, _ in row_data) + HUD_ROW_GAP * (len(row_data) - 1)

    badge_rect = pygame.Rect(HUD_MARGIN, HUD_Y, content_width, content_height).inflate(*HUD_PADDING)

    surface = pygame.display.get_surface()
    badge = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
    badge.fill((0, 0, 0, 235))
    surface.blit(badge, badge_rect.topleft)
    pygame.draw.rect(surface, HUD_BORDER_COLOR, badge_rect, width=2, border_radius=6)

    x = badge_rect.left + HUD_PADDING[0] // 2
    y = badge_rect.top + HUD_PADDING[1] // 2
    surface.blit(header_surface, (x, y))
    y += header_surface.get_height() + HUD_ROW_GAP

    for row_surface, chosen in row_data:
        if chosen:
            highlight_rect = pygame.Rect(x - 3, y - 1, content_width + 6, row_surface.get_height() + 2)
            pygame.draw.rect(surface, HUD_HIGHLIGHT_COLOR, highlight_rect, border_radius=4)
        surface.blit(row_surface, (x, y))
        y += row_surface.get_height() + HUD_ROW_GAP

    # Full update (not just badge_rect) to avoid any partial-rect edge cases under the
    # SCALED fullscreen mode this script runs in.
    pygame.display.update()


def main():
    env = DangerousDaveEnv(render_mode="human", env_rep_type="grid")
    agent = JevAgent()
    hud_font = pygame.font.SysFont(None, HUD_FONT_SIZE, bold=True)
    hud_option_font = pygame.font.SysFont("monospace", HUD_OPTION_FONT_SIZE)

    obs, _ = env.reset()
    episode_reward = 0
    step_count = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        action = agent.act(env)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
        draw_jev_hud(hud_font, hud_option_font, agent.last_decision)
        step_count += 1

        if done or truncated:
            print(f"Episode finished after {step_count} steps. Total reward: {episode_reward:.2f}")
            break

    env.close()


if __name__ == "__main__":
    main()
