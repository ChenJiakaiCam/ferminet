import sys
import site
import os
import re

def patch_kfac_jax():
    site_packages_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        site_packages_dirs.append(site.getusersitepackages())

    for p in sys.path:
        if p not in site_packages_dirs:
            site_packages_dirs.append(p)

    parallel_path = None
    for sp in site_packages_dirs:
        candidate = os.path.join(sp, 'kfac_jax', '_src', 'utils', 'parallel.py')
        if os.path.exists(candidate):
            parallel_path = candidate
            break

    if not parallel_path:
        print("Could not find kfac_jax parallel.py to patch!")
        return

    with open(parallel_path, 'r') as f:
        content = f.read()

    new_code = """
def is_scalar(x: Any) -> bool:
  return isinstance(x, (float, int)) or (
      isinstance(x, jax.Array) and not x.shape
  )

def using_legacy_pmap() -> bool:
  return False

def get_device_n_contents(obj: TArrayTree, n: int) -> TArrayTree:
  def _get_device_n_contents(value: Numeric) -> Numeric:
    if is_scalar(value):
      return value

    if using_legacy_pmap():
      return value[n]

    assert isinstance(value, jax.Array)

    if isinstance(value.sharding, jax.sharding.SingleDeviceSharding):
      return value[n]

    if getattr(jax.sharding, "PositionalSharding", type(None)) is not type(None) and isinstance(value.sharding, getattr(jax.sharding, "PositionalSharding")):
      return value[n]
    if getattr(jax.sharding, "PmapSharding", type(None)) is not type(None) and isinstance(value.sharding, getattr(jax.sharding, "PmapSharding")):
      return value[n]

    NamedSharding = getattr(jax.sharding, "NamedSharding", None) or getattr(jax, "NamedSharding", type(None))
    if not isinstance(value.sharding, NamedSharding):
      return value[n]

    shard_data = value.addressable_shards[n].data

    if len(value.sharding.spec) == 0 or value.sharding.spec[0] is None:
      return shard_data

    if value.shape != shard_data.shape and len(value.shape) == len(shard_data.shape) + 1:
        return shard_data
    elif value.shape == shard_data.shape:
        return shard_data[n]

    if getattr(shard_data, "shape", None) and shard_data.shape[0] == 1:
        return shard_data.squeeze(0)
    return shard_data

  return jax.tree_util.tree_map(_get_device_n_contents, obj)

"""
    start_idx = content.find("def is_scalar(x:")
    end_idx = content.find("def get_first(obj:")
    if start_idx != -1 and end_idx != -1:
        content = content[:start_idx] + new_code + content[end_idx:]

    content = re.sub(
        r'sharding = jax\.NamedSharding\(mesh, jax\.P\(axis_name\)\)',
        r'sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name))',
        content
    )

    content = re.sub(
        r'mesh = jax\.sharding\.Mesh\(devices, \(axis_name,\)\)\n\s*sharding = jax\.NamedSharding\(mesh, jax\.sharding\.PartitionSpec\(axis_name\)\)',
        r'try:\n    mesh = jax.sharding.Mesh(devices, (axis_name,))\n    sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name))\n  except TypeError:\n    import numpy as np\n    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))\n    sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name))',
        content
    )

    content = re.sub(
        r'return jax\.device_put_replicated\(obj, devices=devices\)',
        r'''try:
      return jax.device_put_replicated(obj, devices=devices)
    except AttributeError:
      return jax.device_put(jax.tree_util.tree_map(lambda x: jax.numpy.stack([x] * len(devices)), obj), jax.NamedSharding(jax.sharding.Mesh(__import__('numpy').array(devices), ('d',)), jax.sharding.PartitionSpec('d')))''',
        content
    )

    # Also patch optimizer
    optimizer_path = os.path.join(os.path.dirname(parallel_path), '..', 'optimizer.py')
    if os.path.exists(optimizer_path):
        with open(optimizer_path, 'r') as f:
            opt_content = f.read()

        opt_patch = """      step_counter_first = self.get_first(step_counter)
      import jax
      try:
          return int(step_counter_first)
      except TypeError:
          import numpy as np
          val = np.asarray(step_counter_first)
          if val.size == 1:
              return int(val.item())
          else:
              return int(val[0])"""

        opt_content = re.sub(
            r'      return int\(self\.get_first\(step_counter\)\)',
            opt_patch,
            opt_content
        )

        with open(optimizer_path, 'w') as f:
            f.write(opt_content)

    with open(parallel_path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_kfac_jax()
