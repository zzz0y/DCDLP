import torch

from dcdlp.models.losses import score_routing_losses


def _scores(degree=0.0, cn=0.0, residual=0.0, interaction=0.0):
    values = {
        "degree": torch.as_tensor([degree], dtype=torch.float32),
        "cn": torch.as_tensor([cn], dtype=torch.float32),
        "residual": torch.as_tensor([residual], dtype=torch.float32),
        "interaction": torch.as_tensor([interaction], dtype=torch.float32),
    }
    return {
        "logit": sum(values.values()),
        **{f"score_{key}": value for key, value in values.items()},
    }


def test_cn_routing_loss_is_lower_for_target_dominant_response():
    before = _scores()
    correct = _scores(degree=0.1, cn=2.0, residual=0.1, interaction=0.1)
    wrong = _scores(degree=1.0, cn=0.1, residual=0.5, interaction=0.5)
    correct_route, _ = score_routing_losses(
        before, correct, "cn", margin_cn=0.2
    )
    wrong_route, _ = score_routing_losses(
        before, wrong, "cn", margin_cn=0.2
    )
    assert correct_route.item() < wrong_route.item()


def test_degree_routing_loss_is_lower_for_target_dominant_response():
    before = _scores()
    correct = _scores(degree=2.0, cn=0.1, residual=0.1, interaction=0.1)
    wrong = _scores(degree=0.1, cn=1.0, residual=0.5, interaction=0.5)
    correct_route, _ = score_routing_losses(
        before, correct, "degree", margin_degree=0.2
    )
    wrong_route, _ = score_routing_losses(
        before, wrong, "degree", margin_degree=0.2
    )
    assert correct_route.item() < wrong_route.item()


def test_zero_response_cannot_satisfy_positive_target_floor():
    before = _scores()
    route, target = score_routing_losses(
        before,
        _scores(),
        "cn",
        target_floor_cn=0.5,
    )
    assert (route + target).item() > 0


def test_interaction_route_weight_controls_cross_response_penalty():
    before = _scores()
    after = _scores(cn=1.0, interaction=0.75)
    low, _ = score_routing_losses(
        before, after, "cn", interaction_route_weight=0.0
    )
    high, _ = score_routing_losses(
        before, after, "cn", interaction_route_weight=2.0
    )
    assert high.item() > low.item()

