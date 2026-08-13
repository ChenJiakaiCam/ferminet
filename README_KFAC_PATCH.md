# KFAC-JAX Patch for JAX >= 0.4.30

Due to breaking changes in newer versions of `jax` relating to sharding objects (like `jax.sharding.PositionalSharding` being removed in favor of `NamedSharding`), `kfac-jax` 0.0.8 crashes in `utils/parallel.py`.

This patch automatically updates the code inside your local `site-packages/kfac_jax` directory to successfully run against new JAX versions.

To apply it inside your environment, simply run:

```bash
python patch_kfac_jax.py
```
