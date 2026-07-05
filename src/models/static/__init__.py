# Package marker only. The legacy package-level import
# (`from .llava_wrapper import LlavaStaticVQAModel`, the retired classification-head
# wrapper) was removed in cleanup Pass 2 (2026-07-05) so importing the active engine
# (`from src.models.static.static import StaticPrunedLlava`) no longer pulls retired
# modules into the final-scope runtime. The retired wrapper lives in
# archive/legacy_models/src/models/static/.
