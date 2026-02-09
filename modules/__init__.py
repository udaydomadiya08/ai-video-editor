"""
AI Video Editing Style Learning System - Modules Package

Core modules:
- style_learner: Learns editing style from reference videos
- transition_learner: Learns transition patterns and motion curves
- transition_autoencoder: Neural network for transition embeddings
- music_analyzer: Extracts audio features for beat sync
- music_transition_mapper: Maps music features to transitions
- transition_simulator: Converts embeddings to motion functions
- scene_scorer: Extracts and scores scenes from input video
- auto_editor: Main video generation engine
- smart_cropper: AI-powered 9:16 cropping with subject tracking

Neural transition modules (learn ANY effect from videos):
- transition_frame_extractor: Extracts frame windows around cuts
- neural_transition_vae: VAE that learns actual frame transformations
- neural_effect_generator: Applies learned transitions at generation time
"""
from modules.style_learner import StyleLearner, StyleParameters
from modules.music_analyzer import MusicAnalyzer, MusicFeatures
from modules.scene_scorer import SceneScorer, Scene, ScoredScenes
from modules.auto_editor import AutoEditor

# Neural transition learning modules
from modules.transition_frame_extractor import TransitionFrameExtractor, TransitionSequence
from modules.neural_transition_vae import NeuralTransitionVAE
from modules.neural_effect_generator import NeuralEffectGenerator, TransitionBlender

# Deep style learning modules (Unsupervised video style)
from modules.deep_style_learner import DeepStyleLearner, StylePatch
from modules.neural_style_vae import NeuralStyleVAE

# Universal Effect Learning (Flow + Intensity)
from modules.effect_field_learner import EffectFieldLearner, EffectGrid
from modules.neural_effect_vae import NeuralEffectVAE, UniversalEffectModel


__all__ = [
    'StyleLearner', 'StyleParameters',
    'MusicAnalyzer', 'MusicFeatures',
    'SceneScorer', 'Scene', 'ScoredScenes',
    'AutoEditor',
    # Neural transition learning
    'TransitionFrameExtractor', 'TransitionSequence',
    'NeuralTransitionVAE',
    'NeuralEffectGenerator', 'TransitionBlender',
    # Deep style
    'DeepStyleLearner', 'StylePatch',
    'NeuralStyleVAE',
    # Universal Effects (Flow + Intensity)
    'EffectFieldLearner', 'EffectGrid',
    'NeuralEffectVAE', 'UniversalEffectModel'
]

