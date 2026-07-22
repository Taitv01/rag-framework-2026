"""
Fairy Tale Thumbnail & Master Prompt Engine
===========================================

Generates multi-section, high-CTR Master Prompts for YouTube Thumbnails, Midjourney, DALL-E, Flux, and Veo3.

Features:
- Structured Master Prompt template (Scene composition, Text layout, Border/texture, Style, CTR notes)
- Visual element & transformation mapping (Betel vine, Areca palm, Lime stone, Spirits, Lighting)
- High-CTR text layout generation with 3D gold gradient effects
- Automatic export to prompt files

Usage:
    from src.story.thumbnail_engine import FairyTaleThumbnailEngine

    engine = FairyTaleThumbnailEngine()
    prompt = engine.generate_master_thumbnail_prompt({
        "title_vi": "Sự tích trầu cau",
        "category_badge": "CỔ TÍCH VIỆT NAM",
        "transformation": "three graves sprouting into a betel palm, betel vine, and lime stone",
        "spirits": "two young men and one woman glowing with silver-gold aura"
    })
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class FairyTaleThumbnailEngine:
    """
    Engine for creating master-level high-CTR YouTube thumbnail prompts and image concept templates.
    """

    DEFAULT_STYLE_DIRECTION = (
        "Rich watercolor illustration in traditional Vietnamese watercolor ink-wash style (thủy mặc) "
        "with elegant, flowing ink outlines. Cinematic dramatic dawn lighting — warm golden rays breaking through clouds. "
        "Color palette: lush green betel leaves, warm golden areca trunk, pure white lime glow, soft dawn pinks and oranges, "
        "deep emerald greens, silver-gold spirit auras, earthy browns of village setting. "
        "Highly detailed with emotional, refined Vietnamese features. Premium quality matching high-budget animated fairy tale concept art."
    )

    def generate_master_thumbnail_prompt(self, story_data: Dict[str, Any]) -> str:
        """
        Generate a complete master-level structured YouTube Thumbnail prompt.

        Args:
            story_data: Dictionary containing title, transformation elements, spirits, scene notes

        Returns:
            Structured Master Prompt string
        """
        title_main = story_data.get("title_main", "TRẦU CAU")
        title_prefix = story_data.get("title_prefix", "SỰ TÍCH")
        badge = story_data.get("category_badge", "CỔ TÍCH VIỆT NAM")
        transformation = story_data.get(
            "transformation",
            "three graves sprouting into a majestic areca palm tree (cây cau), lush green betel vines (cây trầu), and gleaming white lime stone (tảng vôi)"
        )
        spirits = story_data.get(
            "spirits",
            "three translucent spirit silhouettes — two young men and one woman — visible faintly in the divine light above, embracing each other peacefully with serene smiles"
        )
        style = story_data.get("style", self.DEFAULT_STYLE_DIRECTION)

        master_prompt = f"""A breathtaking YouTube thumbnail illustration in traditional Vietnamese watercolor ink-wash style (thủy mặc), 16:9 landscape aspect ratio, 1920x1080 resolution, depicting the miraculous transformation moment from the Vietnamese fairy tale "{title_prefix.title()} {title_main.title()}" — {transformation}.

=== SCENE COMPOSITION ===

CENTER (main focal point, 55% of frame): Three miraculous elements rising from three graves in an ethereal glow:
- CENTER: A tall, majestic areca palm tree (cây cau) growing straight and tall from the middle grave, its trunk smooth and elegant, fronds spreading gracefully at the top like a crown. Golden-green light radiates from the trunk.
- LEFT: Lush green betel vines (cây trầu) wrapping tightly around the areca palm trunk, climbing upward with heart-shaped leaves that glow with emerald-green magical light. The vines are vibrant and alive, symbolizing eternal love and devotion.
- RIGHT: A pure white lime stone (tảng vôi) gleaming at the base of the areca palm, radiating soft silver-white light. The lime stone is smooth and luminous, like a precious gem.
All three elements are intertwined, inseparable — representing the three people who could never be separated even in death.

UPPER PORTION: Ethereal golden-white divine light breaking through parting clouds above the three elements, rays of light streaming down like a blessing from heaven. Magical particles and flower petals (lotus, jasmine) drift downward from the light. {spirits}, glowing with silver-gold aura.

LEFT BACKGROUND (20% of frame): A peaceful Vietnamese village at dawn — traditional thatched-roof houses, green rice paddies, bamboo groves. Villagers in the distance pointing at the miraculous scene with expressions of awe and wonder. A winding village road leads toward the viewer.

RIGHT BACKGROUND (20% of frame): A moonlit river scene — weeping willow and cherry blossom trees lining the bank, petals falling gently into the water. A faint ghostly memory of the brothers and the woman by the river, fading into soft mist.

LOWER FOREGROUND: Fresh green grass and wildflowers (marigolds, jasmine) growing around the three graves. Small offerings of betel leaves, areca nuts, and lime arranged neatly in front. A few lit incense sticks with delicate smoke curling upward.

ATMOSPHERIC EFFECTS: Dawn light breaking through after a night of sorrow — the transition from darkness to light symbolizes hope and eternal love. Warm golden-soft morning light bathes the scene. Magical glow emanates from the three elements, creating a divine halo. Fireflies and sparkle particles create a dreamy, sacred atmosphere.

=== TEXT LAYOUT (RENDERED IN IMAGE — HIGH CTR OPTIMIZED) ===

TOP-LEFT (primary title zone, occupying roughly 35% width of frame):
Large bold stacked text:
Line 1: "{title_main.upper()}" — massive 3D extruded metallic gold letters with deep dark-red/maroon drop shadow and warm outer glow. Thick, chunky, highly legible even at small mobile thumbnail size. Polished gold gradient from bright yellow-gold at top to deep amber-gold at bottom.
Line 2: "{title_prefix.upper()}" — smaller but still bold, same gold 3D style, stacked above "{title_main.upper()}" as a prefix.
All Vietnamese diacritics are clearly rendered and properly placed.

BOTTOM-LEFT (subtitle banner):
Bold uppercase text "{badge.upper()}" in warm gold color on a subtle semi-transparent dark ribbon/banner. Clearly readable at all sizes. All Vietnamese diacritics properly rendered.

=== BORDER & TEXTURE ===
Aged parchment/rice paper edges with slight torn watercolor bleeding effects, antique storybook scroll aesthetic. Subtle paper grain texture throughout. Warm dawn glow matching the sacred theme.

=== STYLE DIRECTION ===
{style}

=== CTR OPTIMIZATION NOTES ===
- Unique visual symbol: intertwined cau-trầu-vôi creating instant cultural recognition
- High emotional resonance: glowing spirits in clouds trigger curiosity & nostalgia
- High contrast gold 3D text against atmospheric dawn sky guarantees maximum mobile readability
- Divine lighting & miracle elements boost click-through rate (CTR)
"""
        return master_prompt.strip()

    def export_prompt(self, story_data: Dict[str, Any], output_path: Union[str, Path]) -> str:
        """Export generated master prompt to text file."""
        prompt_str = self.generate_master_thumbnail_prompt(story_data)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(prompt_str, encoding="utf-8")
        return str(out)
