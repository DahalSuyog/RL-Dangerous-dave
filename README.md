# RL-Dangerous-dave

Reinforcement learning applied to a [PyGame](https://www.pygame.org/) re-creation of **Dangerous Dave**, the classic 1990 DOS platformer. The game is wrapped as a [Gymnasium](https://gymnasium.farama.org/) environment and trained with **PPO + Random Network Distillation (RND)** for curiosity-driven exploration.

## Features

- Playable re-creation of Dangerous Dave (`python game.py`).
- Gymnasium environment (`env.py`) with three observation representations:
  - `image` — grayscale screen capture
  - `text` — flattened grid of tile labels
  - `grid` — 2D grid of tile labels
- PPO agent augmented with RND (intrinsic curiosity reward + extrinsic score reward).
- Frame-stacking (4 frames), vectorized environments, and episode statistics.
- Model checkpointing and automated evaluation videos (`ffmpeg`).

## Directory structure

```
.
├── agent.py             # Entry point: train/evaluate the RL agent
├── env.py               # Gymnasium wrapper around the game
├── game.py              # Playable game loop (human play)
├── new.py               # Utility: load a rewards.npy file
├── algo.cfg             # RL algorithm configuration
├── game.cfg             # Game mechanics configuration
├── algos/
│   ├── rnd.py           # RND (PPO + curiosity) implementation
│   └── utils.py         # Wrappers and initialization helpers
├── ddave/               # Dangerous Dave game engine
│   ├── helper.py        # Titles, warp zones, tile loading
│   ├── utils.py         # Core classes: Screen, Map, Player, Tile, etc.
│   ├── levels/          # Level definitions (text-based maps)
│   └── tiles/           # Sprite graphics (game + UI)
├── checkpoint/          # Saved model checkpoints
└── rewards/             # Saved reward arrays and evaluation videos
```

## Dependencies

- Python 3.x
- `pygame`
- `numpy`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `scikit-learn`
- `ffmpeg` (only needed for exporting evaluation videos)

## Installation

```bash
git clone https://github.com/DahalSuyog/RL-Dangerous-dave.git
cd RL-Dangerous-dave
pip install -r requirements.txt
```

Make sure `ffmpeg` is available on your `PATH` if you want evaluation videos.

## Usage

Play the game manually:

```bash
python game.py
```

Test the environment with keyboard input (Esc to quit):

```bash
python env.py
```

Train the RND agent:

```bash
python agent.py --train --model-type rnd
```

Evaluate a trained model:

```bash
python agent.py --evaluate --model-type rnd --model-load-path checkpoint/<model_name>/<update>
```

Watch [Jev](https://typesafe.ai) (TypeSafe AI) play:

```bash
python play_jev.py
```

Watch [Laya](https://huggingface.co/convaiinnovations/laya) (Convai Innovations) play:

```bash
python play_laya.py
```

### Jev (TypeSafe AI)

`play_jev.py` uses [Jev](https://typesafe.ai) to pick Dave's moves in real time, via `jev_agent.py`'s
`JevAgent`. Jev is a text/JSON-in, typed-judgment-out model rather than a vision model, so each decision
`JevAgent` builds a compact description of the tiles near Dave (hazards, solid ground, gaps) and asks Jev
to choose one of the 7 discrete actions from that description.

Requires a `TYPESAFE_API_KEY` (get one at [console.typesafe.ai](https://console.typesafe.ai/)). Copy
`.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

### Laya (Convai Innovations)

`play_laya.py` uses [Laya](https://huggingface.co/convaiinnovations/laya) to pick Dave's moves in real
time, via `laya_agent.py`'s `LayaAgent`. Like Jev, Laya is a text/JSON-in, typed-judgment-out model rather
than a vision model, so each decision `LayaAgent` builds the same compact description of the tiles near
Dave and asks Laya to choose one of the 7 discrete actions from that description.

Unlike Jev, Laya is a local, open-weights model (English checkpoint, ModernBERT-large, 421M parameters) —
no API key is needed. The first run downloads the weights from Hugging Face Hub (one-time, cached
afterward) and inference runs locally on CPU or GPU via the already-installed `torch`.

### `agent.py` arguments

| Flag | Description |
|------|-------------|
| `--train` | Train the model |
| `--evaluate` | Evaluate the model |
| `--model-name` | Name to use for the saved model |
| `--env-rep-type` | `image`, `text`, or `grid` (default `image`) |
| `--model-type` | Model type (only `rnd` currently) |
| `--model-load-path` | Path to a checkpoint to load |

## Configuration

### `game.cfg`

| Key | Description |
|-----|-------------|
| `SCREEN_WIDTH` / `SCREEN_HEIGHT` | Base screen dimensions (scaled by `TILE_SCALE_FACTOR`) |
| `RESCALE_FACTOR` | Downscaling factor for image observations |
| `NUM_OF_LEVELS` | Number of levels |
| `CURRENT_LEVEL` | Starting level |
| `NUM_LIVES` | Player lives |
| `ITEM_SCORE` / `TROPHY_SCORE` / `END_LEVEL_SCORE` | Scoring values |
| `STEP_PENALTY` | Reward applied when no score is gained |
| `EPISODE_TIMESTEPS` | Max timesteps per episode |
| `STICKY_ACTIONS` | Repeat an action over multiple frames |
| `LOCKED_DOOR` | Require the trophy to finish a level |
| `PLAYER_RANDOM_SPAWN` | Randomize the player spawn position |

### `algo.cfg`

| Key | Description |
|-----|-------------|
| `SEED` | Random seed |
| `NUM_ENVS` | Number of parallel environments |
| `NUM_STEPS` | Rollout steps per update |
| `TOTAL_TIMESTEPS` | Total training timesteps |

## Environment

- **Action space**: `Discrete(7)` — `Up`, `Left`, `Right`, `Down`, `Up+Left`, `Up+Right`, `No-op`.
- **Observation space**: depends on `env_rep_type`; `image` mode uses a grayscale `(32, 20)` screen capture (SCREEN_WIDTH/RESCALE_FACTOR × SCREEN_HEIGHT/RESCALE_FACTOR) stacked over 4 frames.
- **Reward**: score delta between steps, with `STEP_PENALTY` applied when no score is gained.

## Algorithm

The agent uses **RND (Random Network Distillation)** combined with PPO:

- A shared CNN encoder produces features for a policy head and **two** value heads — one for extrinsic reward (game score) and one for intrinsic reward (curiosity).
- A fixed random target network and a trainable predictor network compute the curiosity reward as the prediction error on novel observations.
- Extrinsic and intrinsic advantages are computed separately (GAE), combined via `int_coef`/`ext_coef`, and used to optimize the clipped PPO objective.

## Credits

The Dangerous Dave re-creation credits, shown on the title screen, are:

- **Recreated by Arthur, Cattani and Murilo**
- **Professor Leandro K. Wives**
