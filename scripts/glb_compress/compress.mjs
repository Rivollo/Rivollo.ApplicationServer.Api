// Draco-compresses a GLB's mesh geometry using the glTF-Transform API.
//
// Usage:
//   node compress.mjs <input.glb> <output.glb> [gltfOutDir]
//
// Invoked as a subprocess by app/services/glb_compression_service.py. Materials,
// textures, animations, hierarchy, and metadata are untouched — draco() only
// rewrites mesh primitives to use the KHR_draco_mesh_compression extension.
//
// When `gltfOutDir` is supplied, a Draco-compressed glTF PACKAGE is written
// there in addition to the GLB: `model.gltf` plus its external `.bin` and
// texture files. Both outputs are serialised from the SAME in-memory Document
// after a SINGLE draco() transform, which matters for two reasons:
//
//   1. The glTF is produced from the original input GLB, never by converting
//      the compressed GLB. Round-tripping a compressed GLB through a copy would
//      DECODE KHR_draco_mesh_compression and emit uncompressed geometry.
//   2. The GLB and the glTF therefore describe byte-identical geometry and
//      share accessor/material/mesh ordering, so a material index resolved
//      against one file is valid against the other. The colour configurator
//      depends on that: it inspects the GLB while the viewer renders the glTF.

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { draco } from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';
import fs from 'node:fs';
import path from 'node:path';

const [, , inputPath, outputPath, gltfOutDir] = process.argv;

if (!inputPath || !outputPath) {
	console.error('Usage: node compress.mjs <input.glb> <output.glb> [gltfOutDir]');
	process.exit(1);
}

// The entry filename is fixed so the caller knows which file to record as the
// asset URL. Every other file in the package is named by glTF-Transform and
// referenced from this one by RELATIVE uri, so the names must not be rewritten.
const GLTF_ENTRY_NAME = 'model.gltf';

const io = new NodeIO()
	.registerExtensions(ALL_EXTENSIONS)
	.registerDependencies({
		'draco3d.decoder': await draco3d.createDecoderModule(),
		'draco3d.encoder': await draco3d.createEncoderModule(),
	});

const document = await io.read(inputPath);
await document.transform(draco({ method: 'edgebreaker' }));

fs.mkdirSync(path.dirname(path.resolve(outputPath)), { recursive: true });
await io.write(outputPath, document);

if (gltfOutDir) {
	fs.mkdirSync(gltfOutDir, { recursive: true });
	await io.write(path.join(gltfOutDir, GLTF_ENTRY_NAME), document);

	// Fail loudly rather than let an uncompressed package reach storage: the
	// whole point of this path is a COMPRESSED glTF, and a silent regression
	// here would only surface as a slow viewer months later.
	const json = JSON.parse(fs.readFileSync(path.join(gltfOutDir, GLTF_ENTRY_NAME), 'utf8'));
	const primitives = (json.meshes ?? []).flatMap((m) => m.primitives ?? []);
	const compressed = primitives.filter(
		(p) => p.extensions && p.extensions.KHR_draco_mesh_compression,
	);
	if (primitives.length > 0 && compressed.length !== primitives.length) {
		console.error(
			`glTF package is not fully Draco-compressed: ${compressed.length}/${primitives.length} primitives`,
		);
		process.exit(2);
	}

	// Consumed by glb_compression_service.py for logging only.
	console.log(
		JSON.stringify({
			gltf_entry: GLTF_ENTRY_NAME,
			gltf_files: fs.readdirSync(gltfOutDir).sort(),
			draco_primitives: compressed.length,
			total_primitives: primitives.length,
		}),
	);
}
