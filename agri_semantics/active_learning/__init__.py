from agri_semantics.active_learning.active_learners import RandomActiveLearner, BALDActiveLearner
from agri_semantics.constants import ActiveLearners


def get_active_learner(cfg, weights_path, checkpoint_path, active_learner_type: str = None):
    name = active_learner_type if active_learner_type else cfg["active_learning"]["type"]
    if isinstance(cfg, dict):
        if name == ActiveLearners.RANDOM:
            return RandomActiveLearner(cfg, weights_path, checkpoint_path, ActiveLearners.RANDOM)
        elif name == ActiveLearners.BALD:
            return BALDActiveLearner(cfg, weights_path, checkpoint_path, ActiveLearners.BALD)
        else:
            RuntimeError(f"{name} active learner not implemented")
    else:
        raise RuntimeError(f"{type(cfg)} not a valid config")
