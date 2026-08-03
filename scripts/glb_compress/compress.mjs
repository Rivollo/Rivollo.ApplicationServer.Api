// Draco-compresses a GLB's mesh geometry using the glTF-Transform API.
// Invoked as `node compress.mjs <input.glb> <output.glb>` by
// app/services/glb_compression_service.py. Materials, textures, animations,
// hierarchy, and metadata are untouched — draco() only rewrites mesh
// primitives to use the KHR_draco_mesh_compression extension.

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { draco } from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';

const [, , inputPath, outputPath] = process.argv;

if (!inputPath || !outputPath) {
	console.error('Usage: node compress.mjs <input.glb> <output.glb>');
	process.exit(1);
}

const io = new NodeIO()
	.registerExtensions(ALL_EXTENSIONS)
	.registerDependencies({
		'draco3d.decoder': await draco3d.createDecoderModule(),
		'draco3d.encoder': await draco3d.createEncoderModule(),
	});

const document = await io.read(inputPath);
await document.transform(draco({ method: 'edgebreaker' }));
await io.write(outputPath, document);
