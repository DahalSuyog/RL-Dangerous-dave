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
HUD_FONT_SIZE = 32
HUD_MARGIN = 10
HUD_PADDING = (20, 14)
# Sits just below the game's own top overlay bar (score/level/lives), in the plain gameplay
# area, so it doesn't cover any existing UI.
HUD_Y = TOP_OVERLAY_POS * TILE_SCALE_FACTOR + HUD_MARGIN


def draw_jev_hud(font, decision):
    """Draws a bold, high-contrast label showing Jev's latest chosen action, so it's readable
    directly on the game window (and in a recording) instead of only in the console."""
    if decision is None:
        text, color = "JEV: THINKING...", HUD_COLOR_NORMAL
    elif decision["error"]:
        text, color = f"JEV: {decision['action_name'].upper()} - API ERROR, REUSING LAST MOVE", HUD_COLOR_ERROR
    elif decision["low_confidence"]:
        text = f"JEV: {decision['action_name'].upper()}  ({decision['confidence'] * 100:.0f}% CONFIDENCE, UNSURE)"
        color = HUD_COLOR_UNCERTAIN
    else:
        text = f"JEV: {decision['action_name'].upper()}  ({decision['confidence'] * 100:.0f}% CONFIDENCE)"
        color = HUD_COLOR_NORMAL

    text_surface = font.render(text, True, color)
    badge_rect = text_surface.get_rect(topleft=(HUD_MARGIN, HUD_Y)).inflate(*HUD_PADDING)

    surface = pygame.display.get_surface()
    badge = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
    badge.fill((0, 0, 0, 235))
    surface.blit(badge, badge_rect.topleft)
    pygame.draw.rect(surface, HUD_BORDER_COLOR, badge_rect, width=2, border_radius=6)
    surface.blit(text_surface, (badge_rect.left + HUD_PADDING[0] // 2, badge_rect.top + HUD_PADDING[1] // 2))
    # Full update (not just badge_rect) to avoid any partial-rect edge cases under the
    # SCALED fullscreen mode this script runs in.
    pygame.display.update()


def main():
    env = DangerousDaveEnv(render_mode="human", env_rep_type="grid")
    agent = JevAgent()
    hud_font = pygame.font.SysFont(None, HUD_FONT_SIZE, bold=True)

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
        draw_jev_hud(hud_font, agent.last_decision)
        step_count += 1

        if done or truncated:
            print(f"Episode finished after {step_count} steps. Total reward: {episode_reward:.2f}")
            break

    env.close()


if __name__ == "__main__":
    main()
