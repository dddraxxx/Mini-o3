from types import SimpleNamespace

from torch import nn

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class DummyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.patch_embed = nn.Linear(2, 2)
        self.model.visual.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.model.visual.merger = nn.Linear(2, 2)
        self.model.language_model = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 2)


def _engine(*, freeze_vision_tower: bool, freeze_multi_modal_projector: bool):
    engine = object.__new__(FSDPEngine)
    engine.model_config = SimpleNamespace(
        freeze_vision_tower=freeze_vision_tower,
        freeze_multi_modal_projector=freeze_multi_modal_projector,
    )
    return engine


def _all_trainable(module):
    return all(parameter.requires_grad for parameter in module.parameters())


def _all_frozen(module):
    return all(not parameter.requires_grad for parameter in module.parameters())


def test_freeze_vision_tower_keeps_projector_trainable_when_requested():
    model = DummyVLM()

    _engine(freeze_vision_tower=True, freeze_multi_modal_projector=False)._freeze_multimodal_modules(model)

    assert _all_frozen(model.model.visual.patch_embed)
    assert _all_frozen(model.model.visual.blocks)
    assert _all_trainable(model.model.visual.merger)
    assert _all_trainable(model.model.language_model)
    assert _all_trainable(model.lm_head)


def test_freeze_vision_tower_and_projector_keeps_language_trainable():
    model = DummyVLM()

    _engine(freeze_vision_tower=True, freeze_multi_modal_projector=True)._freeze_multimodal_modules(model)

    assert _all_frozen(model.model.visual)
    assert _all_trainable(model.model.language_model)
    assert _all_trainable(model.lm_head)
