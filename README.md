# Skywright UI qualification project

Synthetic RX 7900 XTX workload for Skywright issue #233. No user data or credentials are included.

Proposed repository: `Zorro909/skywright-ui-qualification`, public.
Proposed public GHCR packages: `skywright-ui-qualification-profile` and `skywright-ui-qualification`.

The profile workflow builds the committed Skywright ROCm profile at revision `906b84d17a20bf114db2cf6601fff30be08e974d`. This is a qualification package, not an official Environment Profile release.

After that workflow returns its digest, commit `skywright-project.json` with project identity `skywright-ui-qualification`, repository `ghcr.io/zorro909/skywright-ui-qualification`, configuration `configuration.json`, metrics `metrics.json`, dependency lock `requirements.lock`, smoke command `["python", "-m", "skywright_project"]` and only the `rocm` backend pinned to that exact profile digest. Run the project workflow from the resulting clean commit.

GitHub may initially create the packages as private. An owner must make these two synthetic qualification packages public before the local backend can resolve and pull them without private registry credentials.

The local qualification uses real Dataset publications, actual GPU training, exact checkpoint state and cooperative cancellation through Skywright. CI checks imports and contracts without requiring a GPU. Nothing has been published yet.
