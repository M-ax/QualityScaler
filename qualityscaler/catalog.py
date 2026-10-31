"""Pure application/model catalog, matching the 2026.4 desktop release."""

APP_NAME = "QualityScaler"
VERSION = "2026.4"

MENU_LIST_SEPARATOR = ["----"]
LVA_models = ["LVAx2"]
RealESR_models = ["RealESR_Gx4", "RealESR_Ax4"]
BSRGAN_models = ["BSRGANx2", "BSRGANx4"]
RealESRGAN_models = ["RealESRGANx4"]
MSharp_models = ["MSharpx4"]
IRCNN_models = ["IRCNN_Mx1", "IRCNN_Lx1"]

MODEL_NAMES = tuple(
    LVA_models + RealESR_models + BSRGAN_models
    + RealESRGAN_models + MSharp_models + IRCNN_models
)
AI_models_list = (
    LVA_models + MENU_LIST_SEPARATOR
    + RealESR_models + MENU_LIST_SEPARATOR
    + BSRGAN_models + MENU_LIST_SEPARATOR
    + RealESRGAN_models + MENU_LIST_SEPARATOR
    + MSharp_models + MENU_LIST_SEPARATOR
    + IRCNN_models
)
VRAM_model_usage = {
    "LVAx2": 2,
    "RealESR_Gx4": 2.5,
    "RealESR_Ax4": 2.5,
    "BSRGANx2": 0.8,
    "BSRGANx4": 0.75,
    "RealESRGANx4": 0.75,
    "MSharpx4": 1.5,
    "IRCNN_Mx1": 4,
    "IRCNN_Lx1": 4,
}
