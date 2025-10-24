import os

import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.joint_actions import JointAction
from mjlab.tasks.onnx_export import export_policy_module_to_onnx
from mjlab.third_party.isaaclab.isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_velocity_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
  _export_policy_to_onnx(policy_exporter, path, filename)


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  robot: Entity = env.scene["robot"]
  onnx_path = os.path.join(path, filename)
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointAction)
  ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
  metadata = {
    "run_path": run_path,
    "joint_names": robot.joint_names,
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": env.command_manager.active_terms,
    "observation_names": env.observation_manager.active_terms["policy"],
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
  }

  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)


def _export_policy_to_onnx(
  policy_module: _OnnxPolicyExporter, path: str, filename: str
) -> None:
  os.makedirs(path, exist_ok=True)
  export_path = os.path.join(path, filename)

  dynamic_shapes = _build_dynamic_shapes(policy_module)
  dynamic_axes = _build_dynamic_axes(policy_module)

  if policy_module.is_recurrent:
    obs = torch.zeros(1, policy_module.rnn.input_size)
    h_in = torch.zeros(policy_module.rnn.num_layers, 1, policy_module.rnn.hidden_size)

    if policy_module.rnn_type == "lstm":
      c_in = torch.zeros(policy_module.rnn.num_layers, 1, policy_module.rnn.hidden_size)
      export_policy_module_to_onnx(
        policy_module,
        (obs, h_in, c_in),
        export_path,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
        dynamic_shapes=dynamic_shapes,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        verbose=policy_module.verbose,
      )
    elif policy_module.rnn_type == "gru":
      export_policy_module_to_onnx(
        policy_module,
        (obs, h_in),
        export_path,
        input_names=["obs", "h_in"],
        output_names=["actions", "h_out"],
        dynamic_shapes=dynamic_shapes,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        verbose=policy_module.verbose,
      )
    else:
      raise NotImplementedError(f"Unsupported RNN type: {policy_module.rnn_type}")
  else:
    obs = torch.zeros(1, policy_module.actor[0].in_features)
    export_policy_module_to_onnx(
      policy_module,
      obs,
      export_path,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_shapes=dynamic_shapes,
      dynamic_axes=dynamic_axes,
      opset_version=18,
      verbose=policy_module.verbose,
    )


def _build_dynamic_shapes(
  policy_module: _OnnxPolicyExporter,
) -> tuple[dict[int, "torch.export.Dim"], ...] | None:
  export_mod = getattr(torch, "export", None)
  dim_cls = getattr(export_mod, "Dim", None) if export_mod is not None else None
  if dim_cls is None:
    return None

  batch_dim = dim_cls("batch")
  obs_shape = {0: batch_dim}

  if policy_module.is_recurrent:
    hidden_shape = {1: batch_dim}
    if policy_module.rnn_type == "lstm":
      return (obs_shape, hidden_shape, hidden_shape)
    if policy_module.rnn_type == "gru":
      return (obs_shape, hidden_shape)
    raise NotImplementedError(f"Unsupported RNN type: {policy_module.rnn_type}")

  return (obs_shape,)


def _build_dynamic_axes(policy_module: _OnnxPolicyExporter) -> dict[str, dict[int, str]]:
  dynamic_axes: dict[str, dict[int, str]] = {
    "obs": {0: "batch"},
    "actions": {0: "batch"},
  }
  if policy_module.is_recurrent:
    dynamic_axes["h_in"] = {1: "batch"}
    dynamic_axes["h_out"] = {1: "batch"}
    if policy_module.rnn_type == "lstm":
      dynamic_axes["c_in"] = {1: "batch"}
      dynamic_axes["c_out"] = {1: "batch"}
  return dynamic_axes
