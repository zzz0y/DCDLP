import torch

from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.models.dcdlp import DCDLP
from dcdlp.models.losses import interaction_regularization


def _large_interaction_output():
    score_degree = torch.tensor([0.1])
    score_cn = torch.tensor([0.1])
    score_residual = torch.tensor([0.1])
    score_interaction = torch.tensor([10.0])
    return {
        "logit": score_degree + score_cn + score_residual + score_interaction,
        "score_degree": score_degree,
        "score_cn": score_cn,
        "score_residual": score_residual,
        "score_interaction": score_interaction,
    }


def test_audited_interaction_penalizes_share_but_unrestricted_does_not():
    output = _large_interaction_output()
    audited_share, audited_delta = interaction_regularization(
        output, mode="audited", share_cap=0.2
    )
    unrestricted_share, unrestricted_delta = interaction_regularization(
        output, mode="unrestricted", share_cap=0.2
    )
    assert audited_share.item() > 0
    assert audited_delta.item() == 0
    assert unrestricted_share.item() == 0
    assert unrestricted_delta.item() == 0


def test_audited_interaction_delta_penalty_obeys_cap():
    before = _large_interaction_output()
    after = {key: value.clone() for key, value in before.items()}
    after["score_interaction"] = before["score_interaction"] + 0.5
    after["logit"] = before["logit"] + 0.5
    _, penalty = interaction_regularization(
        before,
        after,
        mode="audited",
        share_cap=1.0,
        delta_cap=0.2,
    )
    assert penalty.item() > 0


def test_disabled_interaction_is_zero_and_parameters_do_not_update():
    dataset = make_smoke_dataset(8)
    model = DCDLP(
        dataset.features.shape[1],
        hidden_dim=16,
        branch_dim=8,
        dropout=0.0,
        interaction_mode="disabled",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.as_tensor(dataset.features)
    edges = torch.as_tensor(dataset.train_pos.T, dtype=torch.long)
    pairs = torch.as_tensor(dataset.test_pos[:3], dtype=torch.long)
    interaction_before = model.interaction.detach().clone()
    scale_before = model.interaction_scale.detach().clone()

    output = model(x, edges, pairs)
    assert torch.count_nonzero(output["score_interaction"]) == 0
    assert torch.count_nonzero(output["interaction_share"]) == 0
    output["logit"].sum().backward()
    assert model.interaction.grad is None
    assert model.interaction_scale.grad is None
    optimizer.step()

    torch.testing.assert_close(model.interaction, interaction_before)
    torch.testing.assert_close(model.interaction_scale, scale_before)

