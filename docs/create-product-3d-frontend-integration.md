# Delta Changes for Frontend: `POST /createProduct`

This file is only for the new changes.
Assumption: frontend is already integrated with the old `createProduct` flow using `multipart/form-data` and already sends:

- `userId`
- `name`
- `image`
- `mask`
- `target_format`
- `asset_id`
- `mesh_asset_id`

## New Frontend Changes Required

Frontend may now additionally send these optional fields:

- `quality`
- `with_mesh_postprocess`
- `with_texture_baking`
- `use_vertex_color`
- `simplify`
- `fill_holes`
- `texture_size`

## Request Contract Changes

No endpoint change.
No content-type change.
Continue using `POST /createProduct` with `FormData`.

## Defaults Updated on Backend

These fields are no longer required from frontend:

- `target_format` now defaults to `glb`
- `asset_id` now defaults to `9`
- `mesh_asset_id` now defaults to `2`

Frontend can keep sending them, but it no longer has to.

## New `quality` Preset Support

Frontend can send:

- `quality=fast`
- `quality=high`
- `quality=max`

If `quality` is invalid, backend returns `400`.

## Backend Preset Behavior

If frontend sends `quality`, backend applies these presets unless overridden explicitly.

### `fast`

- `with_mesh_postprocess=false`
- `with_texture_baking=false`
- `use_vertex_color=true`
- `simplify=0.95`
- `fill_holes=true`
- `texture_size=1024`

### `high`

- `with_mesh_postprocess=true`
- `with_texture_baking=true`
- `use_vertex_color=false`
- `simplify=0.2`
- `fill_holes=true`
- `texture_size=2048`

### `max`

- `with_mesh_postprocess=true`
- `with_texture_baking=true`
- `use_vertex_color=false`
- `simplify=0.0`
- `fill_holes=true`
- `texture_size=4096`

## Override Behavior

If frontend sends `quality` and also sends any of the tuning fields below, explicit values win over the preset:

- `with_mesh_postprocess`
- `with_texture_baking`
- `use_vertex_color`
- `simplify`
- `fill_holes`
- `texture_size`

Example:

- send `quality=high`
- send `texture_size=4096`

Result:

- backend uses `high` preset
- `texture_size` becomes `4096`

## Effective Default Behavior When Frontend Sends Nothing New

If frontend omits `quality` and omits all tuning fields, backend automatically uses effective defaults equal to the `fast` preset.

This means old frontend integration will continue to work without breaking.

## Recommended Frontend Change

If frontend wants only minimal change, do this:

1. Keep the existing request builder.
2. Add optional support for `quality`.
3. Do not add advanced tuning fields unless needed.

## Minimal Code Delta

```ts
if (payload.quality) {
  formData.append("quality", payload.quality);
}
```

## Advanced Optional Fields

If frontend exposes advanced controls, append them as strings because request type is `FormData`:

```ts
if (payload.withMeshPostprocess !== undefined) {
  formData.append("with_mesh_postprocess", String(payload.withMeshPostprocess));
}

if (payload.withTextureBaking !== undefined) {
  formData.append("with_texture_baking", String(payload.withTextureBaking));
}

if (payload.useVertexColor !== undefined) {
  formData.append("use_vertex_color", String(payload.useVertexColor));
}

if (payload.simplify !== undefined) {
  formData.append("simplify", String(payload.simplify));
}

if (payload.fillHoles !== undefined) {
  formData.append("fill_holes", String(payload.fillHoles));
}

if (payload.textureSize !== undefined) {
  formData.append("texture_size", String(payload.textureSize));
}
```

## Suggested Prompt for Frontend Codex

Use this if handing off to another Codex instance:

```text
The frontend is already integrated with POST /createProduct using FormData.
Please make only the delta changes needed for the updated backend contract:
- add optional quality support with values fast | high | max
- optionally support advanced overrides:
  with_mesh_postprocess
  with_texture_baking
  use_vertex_color
  simplify
  fill_holes
  texture_size
- keep backward compatibility with the existing request builder
- target_format, asset_id, and mesh_asset_id may still be sent, but backend now has defaults: glb, 9, 2
- all FormData values must be appended as strings except files
```
