"""Watch Jev (TypeSafe AI) play Dangerous Dave.

Usage:
    python play_jev.py

Requires TYPESAFE_API_KEY to be set (e.g. in a .env file - see .env.example).
"""

import pygame
from dotenv import load_dotenv

from ddave.utils import TILE_SCALE_FACTOR, TOP_OVERLAY_POS
from env import DangerousDaveEnv
from jev_agent import JevAgent

load_dotenv()

HUD_COLOR_NORMAL = (255, 255, 255)
HUD_COLOR_UNCERTAIN = (255, 210, 60)
HUD_COLOR_ERROR = (255, 90, 90)
HUD_MARGIN = 6
# Sits just below the game's own top overlay bar (score/level/lives), in the plain gameplay
# area, so it doesn't cover any existing UI.
HUD_Y = TOP_OVERLAY_POS * TILE_SCALE_FACTOR + HUD_MARGIN


def draw_jev_hud(font, decision):
    """Draws a small translucent label showing Jev's latest chosen action, so it's readable
    directly on the game window instead of only in the console."""
    if decision is None:
        text, color = "Jev: thinking...", HUD_COLOR_NORMAL
    elif decision["error"]:
        text, color = f"Jev: {decision['action_name'].upper()} (API error, reusing last move)", HUD_COLOR_ERROR
    elif decision["low_confidence"]:
        text = f"Jev: {decision['action_name'].upper()}  ({decision['confidence'] * 100:.0f}% confidence, unsure)"
        color = HUD_COLOR_UNCERTAIN
    else:
        text = f"Jev: {decision['action_name'].upper()}  ({decision['confidence'] * 100:.0f}% confidence)"
        color = HUD_COLOR_NORMAL

    text_surface = font.render(text, True, color)
    badge_rect = text_surface.get_rect(topleft=(HUD_MARGIN, HUD_Y)).inflate(10, 6)

    surface = pygame.display.get_surface()
    badge = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
    badge.fill((0, 0, 0, 170))
    surface.blit(badge, badge_rect.topleft)
    surface.blit(text_surface, (badge_rect.left + 5, badge_rect.top + 3))
    pygame.display.update(badge_rect)


def main():
    env = DangerousDaveEnv(render_mode="human", env_rep_type="grid")
    agent = JevAgent()
    hud_font = pygame.font.SysFont(None, 20)

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
